"""Compose authorised query selection and canonical Action-Selector rendering."""

from backend.app.copilot.offline import Language
from backend.app.copilot.refusals import Refusal, guarded_query
from backend.app.copilot.rendering import prepare_turn, select_rendering
from backend.app.copilot.serving import ServingClient
from backend.app.copilot.tools import EvidenceContext
from backend.app.schemas import Answer


def answer_question(
    question: str,
    language: Language,
    context: EvidenceContext,
    authorised_ids: frozenset[str],
    client: ServingClient,
) -> Answer | Refusal:
    """No API-provided claims or model prose are accepted as evidence."""
    result = guarded_query(question, language, context, authorised_ids, client)
    if isinstance(result, Refusal):
        return result
    turn = prepare_turn(result, context, authorised_ids)
    return select_rendering(turn, client, language, "action")
