"""Only the reviewed legacy absence contract is currently published upstream."""

from backend.app.evidence.claims import make_claim, source
from backend.app.schemas import EvidenceClaim, HimaRoadRun


def flow_claim(run: HimaRoadRun) -> EvidenceClaim:
    if run.schema_version not in {None, "1.0"}:
        raise ValueError("flow outputs require a reviewed upstream schema")
    return make_claim(
        run,
        kind="flow_path",
        label="flow_path_unavailable",
        value={"availability": "unavailable", "reason": "unavailable in this HimaRoad run"},
        sources=(source(run, "manifest.json", "/"), source(run, "report.json", "/")),
        method="reviewed-legacy-output-contract-has-no-persisted-flow-path",
        derived=True,
        flags=("flow_path_unavailable",),
    )
