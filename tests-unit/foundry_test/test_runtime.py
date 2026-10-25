import asyncio
import base64
import copy
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from urllib.error import URLError
from urllib.request import urlopen
import wave

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def runtime():
    import torch
    import execution
    import nodes
    from custom_nodes.foundry import nodes as foundry

    foundry.model_management.interrupt_current_processing(False)
    return SimpleNamespace(torch=torch, execution=execution, nodes=nodes, foundry=foundry)


def png(mode="RGB", size=(3, 2), color=None):
    out = BytesIO()
    Image.new(mode, size, color or ("red" if mode == "RGB" else (255, 0, 0, 128))).save(out, format="PNG")
    return out.getvalue()


def image_response(*images):
    return {"data": [{"b64_json": base64.b64encode(data).decode()} for data in images]}


def test_image_tensor_roundtrip_and_transparency(runtime):
    foundry, torch = runtime.foundry, runtime.torch
    batch = foundry.images_to_tensor(image_response(png(), png("RGBA", (2, 1))))
    assert batch.shape == (2, 2, 3, 4)
    assert batch.dtype == torch.float32
    assert batch.device.type == "cpu"
    assert batch[0, 0, 0].tolist() == [1, 0, 0, 1]
    assert batch[1, 0, 0, 3].item() == pytest.approx(128 / 255)
    assert torch.count_nonzero(batch[1, 1]) == 0
    image = Image.open(BytesIO(foundry.tensor_to_png(batch[1])))
    assert image.mode == "RGBA"
    assert image.getpixel((0, 0)) == (255, 0, 0, 128)


def test_mask_polarity(runtime):
    mask = runtime.torch.tensor([[0.0, 1.0], [0.5, 0.0]])
    image = Image.open(BytesIO(runtime.foundry.mask_to_png(mask, 2, 2)))
    assert np.array(image.getchannel("A")).tolist() == [[255, 0], [127, 255]]


@pytest.mark.parametrize("response", [{"data": []}, {"data": [{"url": "https://not-downloaded.example/image"}]}])
def test_invalid_image_response_is_explicit(runtime, response):
    with pytest.raises(RuntimeError, match="Microsoft Foundry"):
        runtime.foundry.images_to_tensor(response)


@pytest.mark.asyncio
async def test_image_generation_preserves_deployment_and_values(runtime, monkeypatch):
    generate = AsyncMock(return_value=image_response(png()))
    monkeypatch.setattr(runtime.foundry.foundry_client, "generate_images", generate)
    result = await runtime.foundry.FoundryImageGenerate.execute(
        "custom-image-name", "sunrise", size="4096x1024", n=25,
    )
    assert result[0].shape == (1, 2, 3, 3)
    generate.assert_awaited_once_with(
        "custom-image-name", prompt="sunrise", quality="auto", size="4096x1024",
        background="auto", n=25, output_format="png",
    )


@pytest.mark.asyncio
async def test_image_edit_does_not_truncate_batched_references(runtime, monkeypatch):
    edit = AsyncMock(return_value=image_response(png()))
    monkeypatch.setattr(runtime.foundry.foundry_client, "edit_images", edit)
    image = runtime.torch.zeros((19, 2, 3, 3))
    mask = runtime.torch.ones((1, 2, 3))
    await runtime.foundry.FoundryImageEdit.execute("deployment", "combine", {"image_0": image}, mask=mask)
    args = edit.await_args.args
    assert len(args[1]) == 19
    assert args[2]["prompt"] == "combine"
    assert Image.open(BytesIO(args[3])).getpixel((0, 0))[3] == 0


@pytest.mark.asyncio
async def test_chat_optional_images_and_preview(runtime, monkeypatch):
    chat = AsyncMock(return_value={"choices": [{"message": {"content": "hello"}}]})
    monkeypatch.setattr(runtime.foundry.foundry_client, "chat", chat)
    result = await runtime.foundry.FoundryChat.execute("deployment", "describe")
    chat.assert_awaited_once_with("deployment", [{"role": "user", "content": [{"type": "text", "text": "describe"}]}], max_completion_tokens=1024)
    assert result[0] == "hello"
    assert result.ui.as_dict() == {"text": ("hello",)}
    await runtime.foundry.FoundryChat.execute(
        "deployment", "describe", system_prompt="Be concise", image=runtime.torch.zeros((2, 2, 3, 3)),
    )
    messages = chat.await_args.args[1]
    assert messages[0] == {"role": "system", "content": "Be concise"}
    assert len(messages[1]["content"]) == 3
    assert messages[1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_chat_refusal_is_an_error(runtime, monkeypatch):
    monkeypatch.setattr(runtime.foundry.foundry_client, "chat", AsyncMock(
        return_value={"choices": [{"message": {"refusal": "content policy"}}]},
    ))
    with pytest.raises(RuntimeError, match="content policy"):
        await runtime.foundry.FoundryChat.execute("deployment", "prompt")


@pytest.mark.asyncio
async def test_transcribe_preserves_all_audio_items(runtime, monkeypatch):
    transcribe = AsyncMock(side_effect=[{"text": "first"}, {"text": "second"}])
    monkeypatch.setattr(runtime.foundry.foundry_client, "transcribe", transcribe)
    audio = {"waveform": runtime.torch.zeros((2, 2, 160)), "sample_rate": 16000}
    result = await runtime.foundry.FoundryTranscribe.execute("asr", audio, language="en")
    assert result[0] == "first\nsecond"
    assert transcribe.await_count == 2
    for call in transcribe.await_args_list:
        assert call.args[0] == "asr"
        assert call.args[2] == "en"
        with wave.open(BytesIO(call.args[1]), "rb") as wav:
            assert (wav.getframerate(), wav.getnchannels(), wav.getnframes()) == (16000, 2, 160)


@pytest.mark.asyncio
async def test_speech_is_standard_cpu_audio(runtime, monkeypatch):
    out = BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(np.zeros(240, dtype="<i2").tobytes())
    monkeypatch.setattr(runtime.foundry.foundry_client, "speech", AsyncMock(return_value=out.getvalue()))
    result = (await runtime.foundry.FoundrySpeech.execute("tts", "hello", "alloy"))[0]
    assert result["sample_rate"] == 24000
    assert result["waveform"].shape == (1, 1, 240)
    assert result["waveform"].dtype == runtime.torch.float32
    assert result["waveform"].device.type == "cpu"


@pytest.mark.asyncio
async def test_interrupt_cancels_inflight_request(runtime, monkeypatch):
    cancelled = []
    checks = 0

    async def pending():
        try:
            await asyncio.sleep(60)
        finally:
            cancelled.append(True)

    def interrupt():
        nonlocal checks
        checks += 1
        if checks > 1:
            raise runtime.foundry.model_management.InterruptProcessingException()

    monkeypatch.setattr(runtime.foundry.model_management, "throw_exception_if_processing_interrupted", interrupt)
    with pytest.raises(runtime.foundry.model_management.InterruptProcessingException):
        await runtime.foundry.await_request(pending())
    assert cancelled == [True]


@pytest.mark.asyncio
async def test_default_graph_executes_and_is_not_reused_next_run(runtime, monkeypatch, tmp_path):
    await runtime.nodes.init_extra_nodes()
    for node_class in runtime.nodes.NODE_CLASS_MAPPINGS.values():
        assert not set(node_class.RETURN_TYPES).intersection({"MODEL", "CLIP", "VAE", "CONDITIONING", "LATENT"})
    assert "CheckpointLoaderSimple" not in runtime.nodes.NODE_CLASS_MAPPINGS
    assert "KSampler" not in runtime.nodes.NODE_CLASS_MAPPINGS
    assert "PrimitiveString" in runtime.nodes.NODE_CLASS_MAPPINGS
    assert "StringFormat" in runtime.nodes.NODE_CLASS_MAPPINGS
    for name in ("comfy.model_patcher", "comfy.model_prefetch", "comfy.sd", "comfy.samplers", "comfy_aimdo", "comfy_kitchen"):
        assert name not in sys.modules
    import folder_paths
    monkeypatch.setattr(folder_paths, "output_directory", str(tmp_path))
    generator = runtime.nodes.NODE_CLASS_MAPPINGS["FoundryImageGenerate"]
    module = sys.modules[generator.__module__]
    generate = AsyncMock(return_value=image_response(png()))
    monkeypatch.setattr(module.foundry_client, "generate_images", generate)
    workflow = json.loads((ROOT / "default_workflow.json").read_text())
    inputs = dict(zip(("deployment", "prompt", "quality", "size", "background", "n"), workflow["nodes"][0]["widgets_values"]))
    prompt = {
        "1": {"class_type": "FoundryImageGenerate", "inputs": inputs},
        "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "Foundry"}},
    }
    valid, error, outputs, _ = await runtime.execution.validate_prompt("first", copy.deepcopy(prompt), None)
    assert valid, error
    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={}, send_sync=Mock())
    executor = runtime.execution.PromptExecutor(
        server, cache_type=runtime.execution.CacheType.CLASSIC, cache_args={"ram": 0, "ram_inactive": 0},
    )
    for prompt_id in ("first", "second"):
        await executor.execute_async(copy.deepcopy(prompt), prompt_id, execute_outputs=outputs)
        assert executor.success, executor.status_messages
    assert generate.await_count == 2
    assert len(list(tmp_path.glob("Foundry*.png"))) == 2
    assert runtime.foundry.model_management.get_torch_device().type == "cpu"
    assert runtime.torch.version.cuda is None
    assert not runtime.torch.cuda.is_initialized()


@pytest.mark.asyncio
async def test_edit_graph_maps_dynamic_images_and_mask(runtime, monkeypatch, tmp_path):
    await runtime.nodes.init_extra_nodes()
    import folder_paths
    monkeypatch.setattr(folder_paths, "output_directory", str(tmp_path))
    edit_class = runtime.nodes.NODE_CLASS_MAPPINGS["FoundryImageEdit"]
    module = sys.modules[edit_class.__module__]
    edit = AsyncMock(return_value=image_response(png()))
    monkeypatch.setattr(module.foundry_client, "edit_images", edit)
    prompt = {
        "1": {"class_type": "EmptyImage", "inputs": {"width": 4, "height": 3, "batch_size": 2, "color": 0}},
        "2": {"class_type": "SolidMask", "inputs": {"value": 1.0, "width": 4, "height": 3}},
        "3": {"class_type": "FoundryImageEdit", "inputs": {
            "deployment": "image-edit", "prompt": "make it red",
            "images.image_0": ["1", 0], "mask": ["2", 0],
        }},
        "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 0], "filename_prefix": "Edit"}},
    }
    valid, error, outputs, _ = await runtime.execution.validate_prompt("edit", copy.deepcopy(prompt), None)
    assert valid, error
    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={}, send_sync=Mock())
    executor = runtime.execution.PromptExecutor(
        server, cache_type=runtime.execution.CacheType.CLASSIC, cache_args={"ram": 0, "ram_inactive": 0},
    )
    await executor.execute_async(prompt, "edit", execute_outputs=outputs)
    assert executor.success, executor.status_messages
    assert len(edit.await_args.args[1]) == 2
    assert Image.open(BytesIO(edit.await_args.args[3])).getpixel((0, 0))[3] == 0
    assert len(list(tmp_path.glob("Edit*.png"))) == 1


def test_server_starts_without_gpu_or_azure_login(tmp_path, unused_tcp_port):
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    log_path = tmp_path / "server.log"
    env = {**os.environ, "FOUNDRY_AUTH": "cli", "FOUNDRY_ENDPOINT": "https://test.openai.azure.com"}
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "main.py", "--port", str(unused_tcp_port), "--disable-auto-launch",
             "--disable-assets", "--user-directory", str(user_dir)],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            base = f"http://127.0.0.1:{unused_tcp_port}"
            deadline = time.monotonic() + 60
            while True:
                assert process.poll() is None, log_path.read_text()
                try:
                    with urlopen(f"{base}/system_stats", timeout=1) as response:
                        stats = json.load(response)
                    break
                except URLError:
                    if time.monotonic() >= deadline:
                        pytest.fail(log_path.read_text())
                    time.sleep(0.1)
            assert {device["type"] for device in stats["devices"]} == {"cpu"}
            with urlopen(f"{base}/object_info", timeout=5) as response:
                node_info = json.load(response)
            assert "FoundryImageGenerate" in node_info
            assert "CheckpointLoaderSimple" not in node_info
            assert "KSampler" not in node_info
            with urlopen(f"{base}/default_workflow", timeout=5) as response:
                assert {node["type"] for node in json.load(response)["nodes"]} == {"FoundryImageGenerate", "SaveImage"}
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
