from __future__ import annotations

from nfl_bets.config import Settings


def test_for_root_ignores_docker_production_paths(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NFL_BETS_DATA_DIR", "/workspace/data")
    monkeypatch.setenv("NFL_BETS_REPORTS_DIR", "/workspace/reports")
    settings = Settings.for_root(tmp_path)
    assert settings.root == tmp_path.resolve()
    assert settings.data_dir == (tmp_path / "data").resolve()
    assert settings.reports_dir == (tmp_path / "reports").resolve()
    assert settings.db_path.is_relative_to(tmp_path.resolve())
