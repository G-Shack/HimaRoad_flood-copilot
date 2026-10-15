"""Model identity is provenance; scientific scores never become impact facts."""

import re

MODEL_FIELDS = frozenset(
    {
        "name",
        "synthetic",
        "checkpoint_sha256",
        "model_version",
        "preprocessing_id",
        "protocol_commit",
        "threshold",
        "operating_threshold",
        "temperature",
        "tile_size",
        "overlap",
        "training_data_versions",
        "calibration_split",
        "torch_version",
    }
)


def validation_prose(value: object) -> bool:
    return isinstance(value, str) and bool(
        re.search(
            r"\b(?:iou|f1|recall|precision|accuracy)\b|human[- ]label|"
            r"himalay.*(?:test|evaluat)|(?:test|evaluat).*himalay",
            value,
            re.I,
        )
    )
