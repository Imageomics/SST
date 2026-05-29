"""Tests for the CLI dispatch and the light-import contract."""

import importlib
import os
import pathlib
import subprocess
import sys

import pytest

from sst.__main__ import SUBCOMMANDS, main

SRC_DIR = str(pathlib.Path(__file__).resolve().parent.parent / "src")


def test_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for name in SUBCOMMANDS:
        assert name in out


def test_no_args_prints_help(capsys):
    # Bare `sst` should print the help listing, not exit silently.
    main([])
    out = capsys.readouterr().out
    assert "usage: sst" in out
    for name in SUBCOMMANDS:
        assert name in out


def test_unknown_command_errors():
    with pytest.raises(SystemExit) as exc:
        main(["does-not-exist"])
    assert exc.value.code != 0


def test_subcommand_modules_import_and_expose_run():
    for module_name, _help in SUBCOMMANDS.values():
        module = importlib.import_module(module_name)
        assert hasattr(module, "add_arguments")
        assert hasattr(module, "run")


def test_import_sst_does_not_pull_in_torch():
    # Importing the package (e.g. to read sst.__version__) must not drag in the deep learning stack.
    code = "import sys, sst; assert 'torch' not in sys.modules; print(sst.__version__)"
    env = {**os.environ, "PYTHONPATH": SRC_DIR}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "2.0.0"
