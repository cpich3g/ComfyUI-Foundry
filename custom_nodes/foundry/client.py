import asyncio
import json as json_module
import os
from typing import Any, Literal, overload
from urllib.parse import urlsplit

import aiohttp
from azure.identity import AzureCliCredential, ChainedTokenCredential, ManagedIdentityCredential

TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"
API_VERSION = "preview"
REQUEST_TIMEOUT = 600


def get_credential() -> AzureCliCredential | ManagedIdentityCredential | ChainedTokenCredential:
    """Azure CLI login or a managed identity, never keys or service principal secrets.

    FOUNDRY_AUTH selects "cli" or "managed_identity"; the default tries the Azure CLI first and then the managed identity.
    AZURE_CLIENT_ID selects a user-assigned managed identity and AZURE_TENANT_ID pins the Azure CLI tenant.
    """
    mode = os.environ.get("FOUNDRY_AUTH", "auto").lower()
    if mode == "cli":
        return AzureCliCredential(tenant_id=os.environ.get("AZURE_TENANT_ID"))
    if mode == "managed_identity":
        return ManagedIdentityCredential(client_id=os.environ.get("AZURE_CLIENT_ID"))
    if mode == "auto":
        return ChainedTokenCredential(
            AzureCliCredential(tenant_id=os.environ.get("AZURE_TENANT_ID")),
            ManagedIdentityCredential(client_id=os.environ.get("AZURE_CLIENT_ID")),
        )
    raise ValueError("FOUNDRY_AUTH must be auto, cli or managed_identity")


def get_endpoint() -> str:
    endpoint = os.environ.get("FOUNDRY_ENDPOINT")
    if not endpoint:
        raise RuntimeError("Set FOUNDRY_ENDPOINT to your Microsoft Foundry resource endpoint, e.g. https://<resource>.services.ai.azure.com")
    parsed = urlsplit(endpoint)
    is_resource = parsed.hostname is not None and parsed.hostname.endswith(
        (".openai.azure.com", ".services.ai.azure.com", ".cognitiveservices.azure.com")
    )
    if (parsed.scheme != "https"
            or not is_resource
            or parsed.port not in (None, 443)
            or parsed.path not in ("", "/")
            or any((parsed.username, parsed.password, parsed.query, parsed.fragment))):
        raise ValueError("FOUNDRY_ENDPOINT must be an HTTPS Microsoft Foundry resource root, not a project endpoint or API path")
    return endpoint.rstrip("/")


async def _headers() -> dict[str, str]:
    def get_token():
        with get_credential() as credential:
            return credential.get_token(TOKEN_SCOPE)

    token = await asyncio.to_thread(get_token)
    return {"Authorization": f"Bearer {token.token}"}


def _error_message(status: int, text: str) -> str:
    try:
        body = json_module.loads(text)
    except json_module.JSONDecodeError:
        body = text
    detail = body.get("error", body) if isinstance(body, dict) else body
    if isinstance(detail, dict):
        detail = detail.get("message", detail)
    return f"Microsoft Foundry request failed ({status}): {detail}"


@overload
async def post(path: str, *, json: dict[str, Any] | None = None, form: aiohttp.FormData | None = None, binary: Literal[False] = False) -> dict[str, Any]: ...


@overload
async def post(path: str, *, json: dict[str, Any] | None = None, form: aiohttp.FormData | None = None, binary: Literal[True]) -> bytes: ...


async def post(path: str, *, json: dict[str, Any] | None = None, form: aiohttp.FormData | None = None, binary: bool = False) -> dict[str, Any] | bytes:
    url = f"{get_endpoint()}{path}"
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Do not replay billable generation requests or forward Entra tokens on redirects.
        async with session.post(url, params={"api-version": API_VERSION}, headers=await _headers(),
                                json=json, data=form, allow_redirects=False) as resp:
            if resp.status >= 300:
                raise RuntimeError(_error_message(resp.status, await resp.text()))
            if binary:
                return await resp.read()
            return await resp.json()


async def generate_images(deployment: str, **params: Any) -> dict[str, Any]:
    return await post("/openai/v1/images/generations", json={"model": deployment, **params})


async def edit_images(deployment: str, images: list[bytes], fields: dict[str, str], mask: bytes | None = None) -> dict[str, Any]:
    form = aiohttp.FormData()
    form.add_field("model", deployment)
    for key, value in fields.items():
        form.add_field(key, value)
    for i, data in enumerate(images):
        form.add_field("image[]", data, filename=f"image_{i}.png", content_type="image/png")
    if mask is not None:
        form.add_field("mask", mask, filename="mask.png", content_type="image/png")
    return await post("/openai/v1/images/edits", form=form)


async def chat(deployment: str, messages: list[dict[str, Any]], **params: Any) -> dict[str, Any]:
    return await post("/openai/v1/chat/completions", json={"model": deployment, "messages": messages, **params})


async def speech(deployment: str, text: str, voice: str) -> bytes:
    return await post("/openai/v1/audio/speech", json={
        "model": deployment, "input": text, "voice": voice, "response_format": "wav",
    }, binary=True)


async def transcribe(deployment: str, audio: bytes, language: str) -> dict[str, Any]:
    form = aiohttp.FormData()
    form.add_field("model", deployment)
    form.add_field("file", audio, filename="audio.wav", content_type="audio/wav")
    form.add_field("response_format", "json")
    if language:
        form.add_field("language", language)
    return await post("/openai/v1/audio/transcriptions", form=form)
