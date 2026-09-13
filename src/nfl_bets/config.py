from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Filesystem and provider configuration; secrets are read only at request time."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="NFL_BETS_",
        extra="ignore",
    )

    root: Path = Field(default_factory=_repository_root)
    data_dir: Path = Field(default_factory=lambda: _repository_root() / "data")
    cache_dir: Path = Field(default_factory=lambda: _repository_root() / "data/cache/nflreadpy")
    raw_dir: Path = Field(default_factory=lambda: _repository_root() / "data/raw")
    curated_dir: Path = Field(default_factory=lambda: _repository_root() / "data/curated")
    runtime_dir: Path = Field(default_factory=lambda: _repository_root() / "data/runtime")
    artifacts_dir: Path = Field(default_factory=lambda: _repository_root() / "artifacts")
    reports_dir: Path = Field(default_factory=lambda: _repository_root() / "reports")
    db_path: Path = Field(
        default_factory=lambda: _repository_root() / "data/runtime/nfl_bets.sqlite3"
    )
    timezone: str = "America/Toronto"
    odds_base_url: str = "https://api.the-odds-api.com"
    odds_sport_key: str = "americanfootball_nfl"
    odds_timeout_seconds: float = 30.0
    odds_freshness_minutes: int = 30
    weekly_call_limit: int = 20
    manual_call_reserve: int = 4
    schedule_match_tolerance_hours: int = 8
    schema_version: str = "1.0.0"
    the_odds_api_key: SecretStr | None = Field(default=None, validation_alias="THE_ODDS_API_KEY")

    def model_post_init(self, __context: object) -> None:
        root = self.root.resolve()
        self.root = root
        if "data_dir" not in self.model_fields_set:
            self.data_dir = root / "data"
        if "cache_dir" not in self.model_fields_set:
            self.cache_dir = self.data_dir / "cache" / "nflreadpy"
        if "raw_dir" not in self.model_fields_set:
            self.raw_dir = self.data_dir / "raw"
        if "curated_dir" not in self.model_fields_set:
            self.curated_dir = self.data_dir / "curated"
        if "runtime_dir" not in self.model_fields_set:
            self.runtime_dir = self.data_dir / "runtime"
        if "artifacts_dir" not in self.model_fields_set:
            self.artifacts_dir = root / "artifacts"
        if "reports_dir" not in self.model_fields_set:
            self.reports_dir = root / "reports"
        if "db_path" not in self.model_fields_set:
            self.db_path = self.runtime_dir / "nfl_bets.sqlite3"
        for field_name in (
            "data_dir",
            "cache_dir",
            "raw_dir",
            "curated_dir",
            "runtime_dir",
            "artifacts_dir",
            "reports_dir",
            "db_path",
        ):
            setattr(self, field_name, getattr(self, field_name).resolve())

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    def ensure_directories(self) -> None:
        for path in (
            self.cache_dir,
            self.raw_dir / "nflverse",
            self.raw_dir / "odds",
            self.curated_dir,
            self.runtime_dir,
            self.artifacts_dir / "models",
            self.reports_dir,
            self.root / "logs",
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
