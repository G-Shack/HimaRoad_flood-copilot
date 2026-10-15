"""All run content is authorised before opening, including symlink targets."""

from pathlib import Path

from backend.app.settings import Settings


class RunContractError(ValueError):
    """An imported run violates the pinned contract or authorised roots."""


def authorised_path(path: Path, settings: Settings, *, root: Path | None = None) -> Path:
    resolved = path.resolve()
    allowed = (root or settings.himaroad_data_dir).resolve()
    reference = (settings.data_root / "reference").resolve()
    if resolved.is_relative_to(reference) or "reference" in {
        part.casefold() for part in resolved.parts
    }:
        raise RunContractError("reference paths are forbidden")
    if not resolved.is_relative_to(allowed):
        raise RunContractError("path is outside the authorised run root")
    return resolved


def check_sources(value) -> None:
    """Reject declared reference sources before hashing/opening any imagery."""
    import re

    if isinstance(value, dict):
        if value.get("purpose") == "evaluation" or value.get("reference_labels"):
            raise RunContractError("evaluation-purpose inputs are forbidden")
        for key, item in value.items():
            if key in {"source", "source_dataset", "provider"} and isinstance(item, str):
                if re.search(
                    r"\b(?:EMSR\d*|UNOSAT|GFM)\b|COPERNICUS\s+(?:EMS\b|EMERGENCY)", item, re.I
                ):
                    raise RunContractError(
                        "validation-only source is forbidden in evidence imports"
                    )
            check_sources(item)
    elif isinstance(value, list):
        for item in value:
            check_sources(item)
