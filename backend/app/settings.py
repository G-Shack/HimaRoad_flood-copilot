"""Explicit roots; environment values and secrets never enter error messages."""

from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", hide_input_in_errors=True, env_ignore_empty=True
    )
    data_root: Path
    himaroad_data_dir: Path
    collab_agent: Literal["codex", "claude"]
    llm_base_url: str | None = None
    llm_model: str | None = None
    report_font: Path | None = None

    @field_validator("data_root", "himaroad_data_dir")
    @classmethod
    def absolute_root(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("root must be absolute")
        return value.resolve()

    @field_validator("llm_base_url")
    @classmethod
    def http_url(cls, value: str | None) -> str | None:
        if value is not None:
            parsed = urlsplit(value)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError("LLM endpoint must be an HTTP URL")
            if parsed.username or parsed.password:
                raise ValueError("credentials must not appear in endpoint URL")
        return value

    @model_validator(mode="after")
    def separate_roots(self) -> "Settings":
        reference = (self.data_root / "reference").resolve()
        if self.himaroad_data_dir.is_relative_to(reference):
            raise ValueError("HimaRoad root must be outside the reference root")
        if bool(self.llm_base_url) != bool(self.llm_model):
            raise ValueError("LLM endpoint and model must be supplied together")
        return self
