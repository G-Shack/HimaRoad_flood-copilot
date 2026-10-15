"""An explicit read-only registry bound to one authorised evidence snapshot."""

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from pydantic import Field, ValidationError

from backend.app.copilot.serving import ServingClient, ToolProposal
from backend.app.evidence import queries
from backend.app.evidence.claims import canonical, verify_claim
from backend.app.schemas import SHA256, Contract, EvidenceClaim
from backend.app.settings import Settings
from backend.app.storage import load_claims


class ToolRejected(ValueError):
    """Fixed control-plane reason; never echo provider arguments or asset names."""


@dataclass(frozen=True, eq=False)
class _ContextSeal:
    claims: tuple[EvidenceClaim, ...]
    evidence_id: str


@dataclass(frozen=True)
class EvidenceContext:
    evidence_id: str
    claims: tuple[EvidenceClaim, ...]
    selected_subject: str | None = None
    _seal: _ContextSeal | None = None


def _verify_context(context: EvidenceContext) -> None:
    runs = {claim.run_id for claim in context.claims}
    ids = sorted(claim.id for claim in context.claims)
    if len(runs) != 1 or len(ids) != len(set(ids)):
        raise ToolRejected("invalid_evidence_context")
    digest = hashlib.sha256(canonical({"run_id": next(iter(runs)), "claim_ids": ids})).hexdigest()
    if digest != context.evidence_id:
        raise ToolRejected("invalid_evidence_context")
    if context._seal is not None:
        if (
            not isinstance(context._seal, _ContextSeal)
            or context._seal.claims is not context.claims
            or context._seal.evidence_id != context.evidence_id
        ):
            raise ToolRejected("invalid_evidence_context")
    else:
        for claim in context.claims:
            try:
                verify_claim(claim)
            except ValueError:
                raise ToolRejected("invalid_evidence_context") from None
    if context.selected_subject is not None and not any(
        claim.kind == "feature" and claim.subject == context.selected_subject
        for claim in context.claims
    ):
        raise ToolRejected("feature_outside_run")


def open_context(
    evidence_id: str,
    authorised_ids: frozenset[str],
    settings: Settings,
    *,
    selected_subject: str | None = None,
) -> EvidenceContext:
    """Authorisation comes from the application, never a model/request allowlist."""
    if evidence_id not in authorised_ids:
        raise ToolRejected("run_not_authorised")
    context = EvidenceContext(evidence_id, load_claims(evidence_id, settings), selected_subject)
    _verify_context(context)
    return context


def open_immutable_context(
    evidence_id: str, authorised_ids: frozenset[str], settings: Settings
) -> EvidenceContext:
    """Verify SQLite completely once, then freeze JSON before retaining a seal.

    Only the server uses this loader; no request/model can supply claims or seals.
    Database/WAL changes invalidate the API cache and require verification again.
    Ordinary contexts retain full mutation/tamper checks on every use.
    """
    from backend.app.copilot.immutable import freeze

    if evidence_id not in authorised_ids:
        raise ToolRejected("run_not_authorised")
    loaded = load_claims(evidence_id, settings)  # Every hash/run/member checked.
    claims = tuple(
        c.model_copy(update={"value": freeze(c.value), "parameters": freeze(c.parameters)})
        for c in loaded
    )
    context = EvidenceContext(evidence_id, claims, _seal=_ContextSeal(claims, evidence_id))
    _verify_context(context)  # Canonical complete membership and seal identity.
    return context


class RunArgs(Contract):
    evidence_id: SHA256


class AssetArgs(RunArgs):
    layer: Literal["roads", "buildings"] = "roads"


class FeatureArgs(RunArgs):
    feature: Literal["selected_asset"] = Field(json_schema_extra={"enum": ["selected_asset"]})


class RouteArgs(FeatureArgs):
    scenario: Literal["baseline", "1", "2", "3", "4"] = "2"
    mode: Literal["vehicle", "pedestrian"] = "vehicle"


# Descriptions are authored code, never data extracted from imported features.
TOOLS = {
    "summarize_run": (RunArgs, "Summarise the selected run's cited observations."),
    "get_hazard_extent": (RunArgs, "Show observed hazard geometry and metric area claims."),
    "list_affected_assets": (AssetArgs, "List potentially affected mapped roads or buildings."),
    "list_potentially_isolated_settlements": (
        RunArgs,
        "List which settlements may be disconnected under the recorded scenario. "
        "Use for a settlement isolation list, "
        "not for a route requested for one selected settlement.",
    ),
    "inspect_feature": (FeatureArgs, "Inspect the application-selected mapped asset."),
    "get_route": (
        RouteArgs,
        "Show the mapped route/path for the application-selected settlement. "
        "Use when asked to show a vehicle or pedestrian route under a named scenario, "
        "even if the settlement may be disconnected. Preserve the requested scenario and mode. "
        "This retrieves hypothetical mapped reachability, never road safety.",
    ),
    "get_repair_priority": (RunArgs, "Show hypothetical single-link reopening priorities."),
    "trace_flow_path": (RunArgs, "Show persisted downstream flow evidence or unavailable."),
    "get_sources_and_limitations": (RunArgs, "Show source acquisitions and limitations."),
    "get_validation_results": (RunArgs, "Show separately loaded validation-only metrics."),
    "get_evidence_validation": (RunArgs, "Show evidence integrity, not model accuracy."),
    "get_primary_care_access": (RunArgs, "Show nullable mapped primary-care reachability."),
    "get_situation_report": (RunArgs, "Select cited summary facts for a situation report."),
}


def _enabled(context: EvidenceContext) -> set[str]:
    names = set(TOOLS)
    if context.selected_subject is None:
        names.discard("inspect_feature")
    if not (context.selected_subject and context.selected_subject.startswith("settlements:")):
        names.discard("get_route")
    return names


def tool_definitions(context: EvidenceContext) -> list[dict]:
    _verify_context(context)
    definitions = []
    for name, (arguments, description) in TOOLS.items():
        if name not in _enabled(context):
            continue
        schema = arguments.model_json_schema()
        if name == "get_route":
            schema["properties"]["scenario"]["enum"] = sorted(
                {c.scenario for c in context.claims if c.kind == "route"}
            )
        schema["properties"]["evidence_id"]["const"] = context.evidence_id
        # Native tool templates may render enum but omit JSON Schema const.
        schema["properties"]["evidence_id"]["enum"] = [context.evidence_id]
        definitions.append(
            {
                "type": "function",
                "function": {"name": name, "description": description, "parameters": schema},
            }
        )
    return definitions


def dispatch(
    proposal: ToolProposal, context: EvidenceContext, authorised_ids: frozenset[str]
) -> queries.QueryResult:
    if context.evidence_id not in authorised_ids:
        raise ToolRejected("run_not_authorised")
    _verify_context(context)
    if proposal.name not in _enabled(context):
        raise ToolRejected("tool_not_allowed")
    arguments = TOOLS[proposal.name][0]
    try:
        args = arguments.model_validate(proposal.arguments, strict=True)
    except ValidationError:
        raise ToolRejected("invalid_tool_arguments") from None
    if args.evidence_id != context.evidence_id:
        raise ToolRejected("run_not_authorised")
    claims = context.claims
    if proposal.name == "list_affected_assets":
        assert isinstance(args, AssetArgs)
        return queries.list_affected_assets(claims, args.layer)
    if proposal.name in {"inspect_feature", "get_route"}:
        assert context.selected_subject is not None
        if proposal.name == "inspect_feature":
            return queries.inspect_feature(claims, context.selected_subject)
        assert isinstance(args, RouteArgs)
        if args.scenario not in {c.scenario for c in claims if c.kind == "route"}:
            raise ToolRejected("invalid_tool_arguments")
        return queries.get_route(
            claims, context.selected_subject.removeprefix("settlements:"), args.scenario, args.mode
        )
    functions = {
        "summarize_run": queries.summarize_run,
        "get_hazard_extent": queries.get_hazard_extent,
        "list_potentially_isolated_settlements": queries.list_potentially_isolated_settlements,
        "get_repair_priority": queries.get_repair_priority,
        "trace_flow_path": queries.trace_flow_path,
        "get_sources_and_limitations": queries.get_sources_and_limitations,
        "get_validation_results": queries.get_validation_results,
        "get_evidence_validation": queries.get_evidence_validation,
        "get_primary_care_access": queries.get_primary_care_access,
        "get_situation_report": queries.summarize_run,
    }
    return functions[proposal.name](claims)


def _question_with_placeholders(question: str, context: EvidenceContext) -> str:
    """Replace known imported names with the application-bound token, or refuse scope.

    This is deliberately conservative: ambiguous names require application selection.
    It is not a geocoder or a claim of exhaustive language/entity recognition.
    """

    def normalise(value: str) -> str:
        text = unicodedata.normalize("NFKC", value).casefold()
        return "".join(c for c in text if unicodedata.category(c) != "Cf")

    text = normalise(question)
    entities: dict[str, set[str]] = {}
    for claim in context.claims:
        if claim.kind != "feature":
            continue
        properties = claim.value["properties"]
        candidates = [claim.subject, *(properties.get(k) for k in ("name", "name:en", "name:ne"))]
        for value in candidates:
            if isinstance(value, str) and value.strip():
                name = normalise(value)
                if name in text:
                    entities.setdefault(name, set()).add(claim.subject)
    # Longest matches first: a hostile full name cannot expose embedded instructions.
    if not entities:
        return text
    pattern = re.compile(
        r"(?<!\w)(?:"
        + "|".join(re.escape(name) for name in sorted(entities, key=lambda n: (-len(n), n)))
        + r")(?!\w)"
    )

    def replace_name(match: re.Match) -> str:
        if entities[match.group()] != {context.selected_subject}:
            raise ToolRejected("feature_outside_run")
        return "selected_asset"

    return pattern.sub(replace_name, text)


def select_query(
    question: str,
    context: EvidenceContext,
    authorised_ids: frozenset[str],
    client: ServingClient,
) -> queries.QueryResult:
    """Transport selection followed by code validation; this is not a chat answer."""
    if context.evidence_id not in authorised_ids:
        raise ToolRejected("run_not_authorised")
    definitions = tool_definitions(context)
    masked_question = _question_with_placeholders(question, context)
    proposal = client.propose_tool(masked_question, definitions)
    return dispatch(proposal, context, authorised_ids)
