import asyncio
import base64
import wave
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from typing_extensions import override

from comfy import model_management
from comfy_api.latest import IO, UI, ComfyExtension
from comfy_extras.nodes_audio import load as load_audio
from . import client as foundry_client


async def await_request(operation):
    task = asyncio.create_task(operation)
    try:
        model_management.throw_exception_if_processing_interrupted()
        while not task.done():
            await asyncio.wait([task], timeout=0.25)
            model_management.throw_exception_if_processing_interrupted()
        return task.result()
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def tensor_to_png(image: torch.Tensor) -> bytes:
    array = np.clip(image.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    out = BytesIO()
    Image.fromarray(array).save(out, format="PNG")
    return out.getvalue()


def mask_to_png(mask: torch.Tensor, width: int, height: int) -> bytes:
    # Comfy masks are 1 where the API expects transparency.
    alpha = Image.fromarray(np.clip((1.0 - mask.cpu().numpy()) * 255.0, 0, 255).astype(np.uint8)).resize((width, height), Image.Resampling.NEAREST)
    rgba = Image.new("RGBA", (width, height), (0, 0, 0, 255))
    rgba.putalpha(alpha)
    out = BytesIO()
    rgba.save(out, format="PNG")
    return out.getvalue()


def images_to_tensor(response: dict) -> torch.Tensor:
    if not response.get("data"):
        raise RuntimeError("Microsoft Foundry returned no images")
    images = []
    for item in response["data"]:
        if not item.get("b64_json"):
            raise RuntimeError("Microsoft Foundry did not return base64 image data; use a GPT Image deployment")
        with Image.open(BytesIO(base64.b64decode(item["b64_json"], validate=True))) as image:
            images.append(image.copy())
    mode = "RGBA" if any("A" in image.getbands() or "transparency" in image.info for image in images) else "RGB"
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    batch = []
    for image in images:
        image = image.convert(mode)
        if image.size != (width, height):
            padded = Image.new(mode, (width, height))
            padded.paste(image, (0, 0))
            image = padded
        batch.append(torch.from_numpy(np.array(image).astype(np.float32) / 255.0))
    return torch.stack(batch)


def image_options() -> list:
    return [
        IO.String.Input("deployment", default="gpt-image-1", tooltip="Name of the image model deployment in your Microsoft Foundry resource."),
        IO.String.Input("prompt", default="", multiline=True),
        IO.Combo.Input("quality", options=["auto", "low", "medium", "high"], default="auto", optional=True),
        IO.String.Input("size", default="1024x1024", optional=True, tooltip="WIDTHxHEIGHT or auto, as supported by your deployment."),
        IO.Combo.Input("background", options=["auto", "opaque", "transparent"], default="auto", optional=True),
        IO.Int.Input("n", default=1, min=1, optional=True, tooltip="How many images to generate."),
    ]


def image_params(quality: str, size: str, background: str, n: int) -> dict:
    return {"quality": quality, "size": size, "background": background, "n": n, "output_format": "png"}


class FoundryNode(IO.ComfyNode):
    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")


class FoundryImageGenerate(FoundryNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="FoundryImageGenerate",
            display_name="Foundry Image Generate",
            category="foundry/image",
            not_idempotent=True,
            description="Generates images with an image model deployed in Microsoft Foundry.",
            inputs=image_options(),
            outputs=[IO.Image.Output()],
        )

    @classmethod
    async def execute(cls, deployment, prompt, quality="auto", size="1024x1024", background="auto", n=1) -> IO.NodeOutput:
        response = await await_request(foundry_client.generate_images(deployment, prompt=prompt, **image_params(quality, size, background, n)))
        return IO.NodeOutput(images_to_tensor(response))


class FoundryImageEdit(FoundryNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="FoundryImageEdit",
            display_name="Foundry Image Edit",
            category="foundry/image",
            not_idempotent=True,
            description="Edits or combines images with an image model deployed in Microsoft Foundry.",
            inputs=[
                *image_options(),
                IO.Autogrow.Input(
                    "images",
                    template=IO.Autogrow.TemplatePrefix(IO.Image.Input("image"), prefix="image_", min=1, max=100),
                    tooltip="Source and reference images.",
                ),
                IO.Mask.Input("mask", optional=True, tooltip="Areas to change in the first image."),
            ],
            outputs=[IO.Image.Output()],
        )

    @classmethod
    async def execute(cls, deployment, prompt, images, quality="auto", size="1024x1024", background="auto", n=1, mask=None) -> IO.NodeOutput:
        sources = [frame for batch in images.values() for frame in batch]
        if not sources:
            raise ValueError("Connect at least one image to Foundry Image Edit")
        mask_png = None
        if mask is not None:
            height, width = sources[0].shape[:2]
            mask_png = mask_to_png(mask[0], width, height)
        fields = {"prompt": prompt, **{key: str(value) for key, value in image_params(quality, size, background, n).items()}}
        response = await await_request(foundry_client.edit_images(deployment, [tensor_to_png(frame) for frame in sources], fields, mask_png))
        return IO.NodeOutput(images_to_tensor(response))


class FoundryChat(FoundryNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="FoundryChat",
            display_name="Foundry Chat",
            category="foundry/text",
            not_idempotent=True,
            is_output_node=True,
            description="Runs a chat model deployed in Microsoft Foundry, optionally with images.",
            inputs=[
                IO.String.Input("deployment", default="gpt-4.1", tooltip="Name of the chat model deployment in your Microsoft Foundry resource."),
                IO.String.Input("prompt", default="", multiline=True),
                IO.String.Input("system_prompt", default="", multiline=True, optional=True),
                IO.Image.Input("image", optional=True),
                IO.Int.Input("max_tokens", default=1024, min=1, optional=True),
            ],
            outputs=[IO.String.Output()],
        )

    @classmethod
    async def execute(cls, deployment, prompt, system_prompt="", image=None, max_tokens=1024) -> IO.NodeOutput:
        content = [{"type": "text", "text": prompt}]
        if image is not None:
            for frame in image:
                data = base64.b64encode(tensor_to_png(frame)).decode()
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}})
        messages = [{"role": "user", "content": content}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        response = await await_request(foundry_client.chat(deployment, messages, max_completion_tokens=max_tokens))
        message = response["choices"][0]["message"]
        if message.get("refusal"):
            raise RuntimeError(f"Microsoft Foundry refused the request: {message['refusal']}")
        text = message.get("content")
        if not isinstance(text, str):
            raise RuntimeError("Microsoft Foundry returned no text; check the deployment and completion token budget")
        return IO.NodeOutput(text, ui=UI.PreviewText(text))


class FoundrySpeech(FoundryNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="FoundrySpeech",
            display_name="Foundry Speech",
            category="foundry/audio",
            not_idempotent=True,
            inputs=[
                IO.String.Input("deployment", default="gpt-4o-mini-tts"),
                IO.String.Input("text", default="", multiline=True),
                IO.String.Input("voice", default="alloy"),
            ],
            outputs=[IO.Audio.Output()],
        )

    @classmethod
    async def execute(cls, deployment, text, voice) -> IO.NodeOutput:
        data = await await_request(foundry_client.speech(deployment, text, voice))
        waveform, sample_rate = load_audio(BytesIO(data))
        return IO.NodeOutput({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate})


class FoundryTranscribe(FoundryNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="FoundryTranscribe",
            display_name="Foundry Transcribe",
            category="foundry/audio",
            not_idempotent=True,
            is_output_node=True,
            inputs=[
                IO.String.Input("deployment", default="gpt-4o-transcribe"),
                IO.Audio.Input("audio"),
                IO.String.Input("language", default="", optional=True, tooltip="Optional ISO language code, such as en."),
            ],
            outputs=[IO.String.Output()],
        )

    @classmethod
    async def execute(cls, deployment, audio, language="") -> IO.NodeOutput:
        waveform = audio["waveform"]
        texts = []
        for item in waveform:
            pcm = np.clip(item.cpu().numpy().T * 32767, -32768, 32767).astype("<i2")
            out = BytesIO()
            with wave.Wave_write(out) as wav:
                wav.setnchannels(item.shape[0])
                wav.setsampwidth(2)
                wav.setframerate(audio["sample_rate"])
                wav.writeframes(pcm.tobytes())
            response = await await_request(foundry_client.transcribe(deployment, out.getvalue(), language))
            texts.append(response["text"])
        text = "\n".join(texts)
        return IO.NodeOutput(text, ui=UI.PreviewText(text))


class FoundryExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [FoundryImageGenerate, FoundryImageEdit, FoundryChat, FoundrySpeech, FoundryTranscribe]


async def comfy_entrypoint() -> FoundryExtension:
    return FoundryExtension()
