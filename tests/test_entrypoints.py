from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_defines_cross_platform_console_script_entrypoint():
    pyproject_path = ROOT / "pyproject.toml"
    assert pyproject_path.exists()

    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    scripts = data["project"]["scripts"]

    assert scripts["openai-image-generator"] == "generate_image:main"
    assert scripts["generate-image"] == "generate_image:main"


def test_macos_launcher_exists_and_invokes_python_cli():
    launcher = ROOT / "scripts" / "generate-image.command"
    text = launcher.read_text(encoding="utf-8")

    assert text.startswith("#!/usr/bin/env bash")
    assert "generate_image.py" in text
    assert '"$@"' in text


def test_windows_batch_launcher_exists_and_invokes_python_cli():
    launcher = ROOT / "scripts" / "generate-image.bat"
    text = launcher.read_text(encoding="utf-8")

    assert "@echo off" in text.lower()
    assert "generate_image.py" in text
    assert "%*" in text
