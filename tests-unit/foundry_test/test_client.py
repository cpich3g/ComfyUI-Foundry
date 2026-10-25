import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, Mock

from aiohttp import web
from azure.core.credentials import AccessToken
import pytest


@pytest.fixture
def client():
    path = Path(__file__).resolve().parents[2] / "custom_nodes" / "foundry" / "client.py"
    spec = importlib.util.spec_from_file_location("foundry_client_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("host", ["test.openai.azure.com", "test.services.ai.azure.com", "test.cognitiveservices.azure.com"])
def test_resource_root(client, monkeypatch, host):
    monkeypatch.setenv("FOUNDRY_ENDPOINT", f"https://{host}/")
    assert client.get_endpoint() == f"https://{host}"


@pytest.mark.parametrize("endpoint", [
    "http://test.openai.azure.com",
    "https://openai.azure.com",
    "https://test.openai.azure.com.attacker.example",
    "https://attacker.example",
    "https://test.openai.azure.com:8080",
    "https://user:password@test.openai.azure.com",
    "https://test.openai.azure.com/openai/v1",
    "https://test.services.ai.azure.com/api/projects/test",
    "https://test.openai.azure.com/?query=value",
    "https://test.openai.azure.com/#fragment",
])
def test_reject_non_resource_endpoints(client, monkeypatch, endpoint):
    monkeypatch.setenv("FOUNDRY_ENDPOINT", endpoint)
    with pytest.raises(ValueError, match="resource root"):
        client.get_endpoint()


def test_endpoint_required(client, monkeypatch):
    monkeypatch.delenv("FOUNDRY_ENDPOINT", raising=False)
    with pytest.raises(RuntimeError, match="FOUNDRY_ENDPOINT"):
        client.get_endpoint()


@pytest.mark.parametrize("mode", ["cli", "managed_identity", "auto"])
def test_only_cli_and_managed_identity_credentials(client, monkeypatch, mode):
    cli, managed, chain = Mock(), Mock(), Mock()
    monkeypatch.setattr(client, "AzureCliCredential", cli)
    monkeypatch.setattr(client, "ManagedIdentityCredential", managed)
    monkeypatch.setattr(client, "ChainedTokenCredential", chain)
    monkeypatch.setenv("FOUNDRY_AUTH", mode)
    monkeypatch.setenv("AZURE_TENANT_ID", "test-tenant")
    monkeypatch.setenv("AZURE_CLIENT_ID", "test-managed-identity")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "unused-test-secret")
    credential = client.get_credential()
    if mode in ("cli", "auto"):
        cli.assert_called_once_with(tenant_id="test-tenant")
    else:
        cli.assert_not_called()
    if mode in ("managed_identity", "auto"):
        managed.assert_called_once_with(client_id="test-managed-identity")
    else:
        managed.assert_not_called()
    if mode == "auto":
        chain.assert_called_once_with(cli.return_value, managed.return_value)
        assert credential is chain.return_value
    else:
        chain.assert_not_called()


def test_default_credential_chain(client, monkeypatch):
    monkeypatch.delenv("FOUNDRY_AUTH", raising=False)
    monkeypatch.setattr(client, "AzureCliCredential", cli := Mock())
    monkeypatch.setattr(client, "ManagedIdentityCredential", managed := Mock())
    monkeypatch.setattr(client, "ChainedTokenCredential", chain := Mock())
    client.get_credential()
    chain.assert_called_once_with(cli.return_value, managed.return_value)


def test_no_key_or_service_principal_auth_mode(client, monkeypatch):
    monkeypatch.setenv("FOUNDRY_AUTH", "service_principal")
    with pytest.raises(ValueError, match="cli or managed_identity"):
        client.get_credential()


@pytest.mark.asyncio
async def test_token_scope_and_credential_cleanup(client, monkeypatch):
    credential = Mock()
    credential.get_token.return_value = AccessToken("test-token", 9999999999)
    context = Mock()
    context.__enter__ = Mock(return_value=credential)
    context.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(client, "get_credential", lambda: context)
    assert await client._headers() == {"Authorization": "Bearer test-token"}
    credential.get_token.assert_called_once_with("https://cognitiveservices.azure.com/.default")
    context.__exit__.assert_called_once()


@pytest.mark.asyncio
async def test_generation_request(client, monkeypatch, aiohttp_server):
    received = []

    async def generate(request):
        received.append((await request.json(), request.headers["Authorization"], dict(request.query)))
        return web.json_response({"data": [{"b64_json": "test-image"}]})

    app = web.Application()
    app.router.add_post("/openai/v1/images/generations", generate)
    server = await aiohttp_server(app)
    monkeypatch.setattr(client, "get_endpoint", lambda: str(server.make_url("")).rstrip("/"))
    monkeypatch.setattr(client, "_headers", AsyncMock(return_value={"Authorization": "Bearer test-token"}))
    result = await client.generate_images("my-custom-deployment", prompt="landscape", n=2)
    assert result == {"data": [{"b64_json": "test-image"}]}
    assert received == [
        ({"model": "my-custom-deployment", "prompt": "landscape", "n": 2},
         "Bearer test-token", {"api-version": "preview"})
    ]


@pytest.mark.asyncio
async def test_edit_multipart(client, monkeypatch, aiohttp_server):
    received = []

    async def edit(request):
        reader = await request.multipart()
        async for part in reader:
            received.append((part.name, part.filename, bytes(await part.read())))
        return web.json_response({"data": []})

    app = web.Application()
    app.router.add_post("/openai/v1/images/edits", edit)
    server = await aiohttp_server(app)
    monkeypatch.setattr(client, "get_endpoint", lambda: str(server.make_url("")).rstrip("/"))
    monkeypatch.setattr(client, "_headers", AsyncMock(return_value={}))
    await client.edit_images("image-deployment", [b"first", b"second"], {"prompt": "combine", "n": "2"}, b"mask")
    assert received == [
        ("model", None, b"image-deployment"),
        ("prompt", None, b"combine"),
        ("n", None, b"2"),
        ("image[]", "image_0.png", b"first"),
        ("image[]", "image_1.png", b"second"),
        ("mask", "mask.png", b"mask"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [302, 400, 401, 403, 429, 500, 503])
async def test_errors_are_not_retried_or_redirected(client, monkeypatch, aiohttp_server, status):
    requests = []

    async def fail(request):
        requests.append(request.path)
        return web.json_response({"error": {"message": "provider failure"}}, status=status, headers={"Location": "/unexpected"})

    async def unexpected(request):
        requests.append(request.path)
        return web.json_response({})

    app = web.Application()
    app.router.add_post("/openai/v1/chat/completions", fail)
    app.router.add_route("*", "/unexpected", unexpected)
    server = await aiohttp_server(app)
    monkeypatch.setattr(client, "get_endpoint", lambda: str(server.make_url("")).rstrip("/"))
    monkeypatch.setattr(client, "_headers", AsyncMock(return_value={}))
    with pytest.raises(RuntimeError, match=f"\\({status}\\): provider failure"):
        await client.chat("chat-deployment", [{"role": "user", "content": "hello"}])
    assert requests == ["/openai/v1/chat/completions"]


@pytest.mark.asyncio
async def test_non_json_error(client, monkeypatch, aiohttp_server):
    async def fail(request):
        return web.Response(text="gateway unavailable", status=502)

    app = web.Application()
    app.router.add_post("/openai/v1/chat/completions", fail)
    server = await aiohttp_server(app)
    monkeypatch.setattr(client, "get_endpoint", lambda: str(server.make_url("")).rstrip("/"))
    monkeypatch.setattr(client, "_headers", AsyncMock(return_value={}))
    with pytest.raises(RuntimeError, match="gateway unavailable"):
        await client.chat("deployment", [])


@pytest.mark.asyncio
async def test_speech_and_transcription(client, monkeypatch, aiohttp_server):
    received = []

    async def speech(request):
        received.append(await request.json())
        return web.Response(body=b"test-wave-file", content_type="audio/wav")

    async def transcribe(request):
        parts = []
        async for part in await request.multipart():
            parts.append((part.name, bytes(await part.read())))
        received.append(parts)
        return web.json_response({"text": "hello"})

    app = web.Application()
    app.router.add_post("/openai/v1/audio/speech", speech)
    app.router.add_post("/openai/v1/audio/transcriptions", transcribe)
    server = await aiohttp_server(app)
    monkeypatch.setattr(client, "get_endpoint", lambda: str(server.make_url("")).rstrip("/"))
    monkeypatch.setattr(client, "_headers", AsyncMock(return_value={}))
    assert await client.speech("voice-deployment", "hello", "alloy") == b"test-wave-file"
    assert await client.transcribe("asr-deployment", b"input-wave", "en") == {"text": "hello"}
    assert received == [
        {"model": "voice-deployment", "input": "hello", "voice": "alloy", "response_format": "wav"},
        [("model", b"asr-deployment"), ("file", b"input-wave"), ("response_format", b"json"), ("language", b"en")],
    ]
