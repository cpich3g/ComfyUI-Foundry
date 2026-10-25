# ComfyUI Foundry

A CPU-only fork of [ComfyUI](https://github.com/Comfy-Org/ComfyUI) for inference on Microsoft Foundry. The local process runs the graph editor, queue, image/audio/video processing and file output. It does not load or run model weights.

The bundled `custom_nodes/foundry` extension owns remote inference and authentication. Other providers, local checkpoint/sampler/VAE/CLIP nodes, arbitrary custom-node discovery and GPU startup options are not enabled. Existing local-model workflows are **not compatible**; there is no automatic conversion of a diffusion graph into a hosted model request.

## Install

Use Python 3.12 or 3.13 in a new virtual environment:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
```

Linux and Windows explicitly require CPU-only PyTorch wheels. macOS uses the standard PyTorch wheel, but the host still uses CPU tensors only. CUDA, NVIDIA libraries, ROCm, torchvision, torchsde, Transformers, Comfy Kitchen and AIMDO are not runtime dependencies.

## Configure Microsoft Foundry

Deploy the models you need in your own Foundry resource. Set the **resource root**, not a project URL or a URL containing `/openai/v1`:

```bash
export FOUNDRY_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com
```

Resource roots under `services.ai.azure.com` and `cognitiveservices.azure.com` are also accepted. Only public-Azure HTTPS resource endpoints are supported. Each node takes a **deployment name**, which may differ from the model's catalog name.

On PowerShell, set environment variables with `$env:FOUNDRY_ENDPOINT = "https://YOUR-RESOURCE.openai.azure.com"`.

### Azure CLI authentication

Install the [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli), then:

```bash
az login --tenant YOUR-TENANT-ID
export AZURE_TENANT_ID=YOUR-TENANT-ID
export FOUNDRY_AUTH=cli
```

Pin the tenant when the Foundry resource is not in your CLI's default tenant. No Azure subscription discovery or resource provisioning happens automatically.

### Managed identity

On an Azure host with a system-assigned managed identity:

```bash
export FOUNDRY_AUTH=managed_identity
```

For a user-assigned managed identity, also set `AZURE_CLIENT_ID` to its client ID. The identity must be attached to the Azure host. `AZURE_CLIENT_ID` is **not** used to authenticate a service principal.

If `FOUNDRY_AUTH` is omitted, the extension tries `AzureCliCredential` followed by `ManagedIdentityCredential`. No API keys, client secrets, service-principal credentials, browser login, or `DefaultAzureCredential` fallback are used.

Grant the signed-in user or managed identity inference access on the resource, typically **Cognitive Services OpenAI User** for Azure OpenAI deployments. Authentication failures and provider errors are surfaced in the workflow, not replaced by local inference.

## Run

```bash
python main.py
```

Open `http://127.0.0.1:8188`. The default workflow is **Foundry Image Generate -> Save Image**. Set its deployment to your image deployment, enter a prompt and queue the workflow. No local weights or GPU flags are needed.

| Node | Hosted API | Output |
|------|------------|--------|
| Foundry Image Generate | GPT Image generations | IMAGE |
| Foundry Image Edit | GPT Image edits, reference images and optional mask | IMAGE |
| Foundry Chat | OpenAI-compatible chat completions, optional image input | STRING with text preview |
| Foundry Speech | OpenAI-compatible speech synthesis | AUDIO |
| Foundry Transcribe | OpenAI-compatible audio transcription | STRING with text preview |

Connect image outputs to the native Save Image or Preview Image nodes, and audio outputs to native Save Audio or Preview Audio nodes. Chat and transcription can run as output nodes without an additional consumer. Native text, math and logic nodes remain available for building workflows. Image edit masks use ComfyUI's convention: white marks areas to change.

The extension uses `/openai/v1` APIs with `api-version=preview`. Deployments must support the selected API and its parameters. Image nodes require base64 image responses, as returned by GPT Image models; URL-only responses are not downloaded. Local video tools handle media files but do not generate video remotely. Other Foundry modalities are not implemented by these nodes.

Generation is not reused from a previous queue run. Billable requests are not automatically retried, and HTTP redirects are not followed. Credentials stay on the server and are not stored in workflows. Prompts and connected media are sent to your configured Foundry resource only when a Foundry node executes.

The server defaults to localhost. Anyone who can queue workflows uses the server's Azure identity and can incur charges; restrict access before exposing it to other users. Additional custom-node packages are intentionally not loaded. `--disable-all-custom-nodes`, `--disable-partner-nodes` or `--offline` can disable the Foundry extension.

## Development

```bash
python -m pip install pytest pytest-asyncio pytest-aiohttp ruff
python -m pytest tests-unit/foundry_test tests-unit/governance_test/test_initialize.py
ruff check .
```

Foundry tests use mocked credentials and a local HTTP server, not a paid deployment. Upstream tests for local model loading, GPU kernels and samplers no longer describe this fork's runtime.

Upstream ComfyUI copyright notices and the [GPL-3.0 license](LICENSE) are retained.
