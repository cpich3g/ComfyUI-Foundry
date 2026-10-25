import ast
import json
from pathlib import Path

from packaging.requirements import Requirement
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("platform", ["linux", "win32", "darwin"])
def test_cpu_dependency_selection(platform):
    requirements = [
        Requirement(line) for line in (ROOT / "requirements.txt").read_text().splitlines()
        if line and not line.startswith(("#", "--"))
    ]
    selected = [r for r in requirements if not r.marker or r.marker.evaluate({"sys_platform": platform})]
    names = {r.name for r in selected}
    assert not names.intersection({"torchvision", "torchsde", "comfy-kitchen", "comfy-aimdo", "transformers", "spandrel"})
    torch = next(r for r in selected if r.name == "torch")
    assert str(torch.specifier) == ("==2.9.1" if platform == "darwin" else "==2.9.1+cpu")


def test_default_workflow_is_foundry_generation_and_save():
    workflow = json.loads((ROOT / "default_workflow.json").read_text())
    assert {n["type"] for n in workflow["nodes"]} == {"FoundryImageGenerate", "SaveImage"}
    generator = next(n for n in workflow["nodes"] if n["type"] == "FoundryImageGenerate")
    output = next(n for n in workflow["nodes"] if n["type"] == "SaveImage")
    assert workflow["links"] == [[1, generator["id"], 0, output["id"], 0, "IMAGE"]]
    assert generator["widgets_values"][0] == "gpt-image-1"


def test_startup_has_no_gpu_initialization_or_external_prestartup():
    tree = ast.parse((ROOT / "main.py").read_text())
    functions = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    assert len(functions) == len(set(functions))
    assert "execute_prestartup_script" not in functions
    assert "cuda_malloc_warning" not in functions
    source = (ROOT / "main.py").read_text()
    assert "comfy_aimdo" not in source
    assert "cuda_malloc" not in source


def test_active_runtime_cli_attributes_exist():
    from comfy.cli_args import args

    for relative in ("main.py", "nodes.py", "server.py", "execution.py", "comfy/utils.py"):
        tree = ast.parse((ROOT / relative).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "args":
                assert hasattr(args, node.attr), f"{relative}:{node.lineno}: args.{node.attr}"


@pytest.mark.parametrize("flag", ["--cuda-device", "--gpu-only", "--directml", "--enable-triton-backend", "--enable-dynamic-vram"])
def test_gpu_flags_are_not_advertised(flag):
    assert f'"{flag}"' not in (ROOT / "comfy" / "cli_args.py").read_text()
