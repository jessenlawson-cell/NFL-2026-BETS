from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from nfl_bets.challenger import _verify_challenger_git_identity
from nfl_bets.prospective import ProspectiveDataError

PROTECTED = ("challenger.py",)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "challenger.py").write_text("frozen\n", encoding="utf-8")
    _git(tmp_path, "add", "challenger.py")
    _git(tmp_path, "commit", "-m", "freeze")
    return tmp_path, _git(tmp_path, "rev-parse", "HEAD")


def test_git_identity_accepts_descendant_with_unchanged_protected_files(tmp_path: Path) -> None:
    root, freeze_commit = _repository(tmp_path)
    (root / "record.json").write_text("{}\n", encoding="utf-8")
    _git(root, "add", "record.json")
    _git(root, "commit", "-m", "record freeze")

    assert _verify_challenger_git_identity(root, freeze_commit, PROTECTED) == _git(
        root, "rev-parse", "HEAD"
    )


def test_git_identity_rejects_descendant_with_changed_protected_files(tmp_path: Path) -> None:
    root, freeze_commit = _repository(tmp_path)
    (root / "challenger.py").write_text("changed\n", encoding="utf-8")
    _git(root, "add", "challenger.py")
    _git(root, "commit", "-m", "change model")

    with pytest.raises(ProspectiveDataError, match="changed since challenger freeze"):
        _verify_challenger_git_identity(root, freeze_commit, PROTECTED)
