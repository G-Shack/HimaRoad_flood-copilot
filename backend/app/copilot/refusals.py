"""Conservative code-level refusals, before any model or query operation."""

import re
import unicodedata
from typing import Literal

from backend.app.copilot.offline import NATIVE_REVIEW, NOTICE, Language
from backend.app.copilot.serving import LLMUnavailable, ServingClient
from backend.app.copilot.tools import EvidenceContext, ToolRejected, select_query
from backend.app.evidence.queries import QueryResult
from backend.app.schemas import Contract

RefusalReason = Literal[
    "casualties",
    "depth",
    "confirmed_condition",
    "safe_route",
    "cause",
    "forecast",
    "out_of_run",
    "unsupported",
    "evidence_unavailable",
    "llm_unavailable",
]

# Conservative screening is not a claim of complete EN/NE paraphrase coverage.
PATTERNS: tuple[tuple[RefusalReason, str], ...] = (
    (
        "casualties",
        r"\b(?:death\w*|dead|died|die|killed|fatal\w*|casualt\w*|perish\w*|"
        r"missing|unaccounted|injur\w*|surviv\w*|alive|people|persons|population)\b|"
        r"मृत|मरे|मृत्यु|ज्यान|हताहत|बेपत्ता|हराएका|घाइते|मानिस|व्यक्ति|जनसंख्या|बाँचेक",
    ),
    (
        "depth",
        r"\b(?:depth|deep|water\s+level|how\s+high)\b|गहिर|जलस्तर|पानीको\s*उचाइ|"
        r"पानी.*कति.*मिटर",
    ),
    (
        "confirmed_condition",
        r"\b(?:collaps\w*|destroy\w*|demolish\w*|closed|open)\b|"
        r"पक्का.*(?:क्षति|भत्क|बन्द)|भत्क|ध्वस्त|बन्द\s*(?:छ|भए)|खुला\s*छ",
    ),
    (
        "safe_route",
        r"\b(?:saf\w*|passable|risk.free|without\s+risk|usable)\b|"
        r"सुरक्षित|सुरक्षा|जोखिमरहित|(?:जान|हिँड्न|चल्न)\s*मिल्छ",
    ),
    (
        "cause",
        r"\b(?:cause\w*|glof|trigger\w*)\b|why.*(?:happen|occur)|how.*(?:flood|event).*(?:start|begin)|"
        r"glacial\s+lake|कारण|हिमताल|किन.*(?:आयो|आएको|बाढी)|कसरी.*सुरु",
    ),
    (
        "forecast",
        r"\b(?:forecast\w*|predict\w*|tomorrow|future)\b|next\s+(?:day|week|month)|"
        r"will.*(?:flood|water|rain|rise|recede|reach)|कहिले.*घट|भोलि|भविष्य|पूर्वानुमान|अर्को\s*हप्ता",
    ),
    (
        "out_of_run",
        r"\b(?:elsewhere|outside|another\s+(?:run|area|country)|other\s+country)\b|"
        r"बाहिर|अर्को\s*(?:रन|क्षेत्र|देश)|अन्य\s*देश",
    ),
)
SUPPORTED = (
    r"\b(?:run|summar\w*|report|scene\w*|source\w*|observ\w*|uncertain\w*|limitation\w*|"
    r"satellite|imagery|flood|debris|hazard|extent|reach\w*|affected|exposed|building\w*|"
    r"road\w*|bridge\w*|settlement\w*|village\w*|town|hospital|care|route\w*|path|link\w*|"
    r"scenario|repair|reconnect\w*|downstream|validation|integrity|evidence)\b|"
    r"रन|सारांश|प्रतिवेदन|रिपोर्ट|दृश्य|स्रोत|अनिश्चित|सीमा|उपग्रह|बाढी|गेग्रान|भवन|सडक|पुल|"
    r"बस्ती|गाउँ|अस्पताल|मार्ग|सम्पर्क|परिदृश्य|मर्मत|बहाव|तल्लो|प्रमाण|अखण्डता"
)
SELECTED_ASSET = (
    r"\bselected[\s_-]+(?:assets?|settlements?|features?|roads?|buildings?|bridges?|villages?)\b|"
    r"छानिएको\s*(?:सम्पत्ति|बस्ती|वस्तु|सडक|भवन|पुल|गाउँ)"
)
TEXT: dict[RefusalReason, tuple[str, str]] = {
    "casualties": (
        "Satellite-map evidence cannot establish casualty or population figures.",
        "उपग्रह नक्साको प्रमाणबाट हताहत वा जनसंख्याको संख्या निर्धारण गर्न सकिँदैन।",
    ),
    "depth": (
        "These observations do not establish water depth or water level.",
        "यी अवलोकनले पानीको गहिराइ वा जलस्तर निर्धारण गर्दैनन्।",
    ),
    "confirmed_condition": (
        "These maps cannot confirm damage, collapse, or operational road condition.",
        "यी नक्साले क्षति, संरचना भत्किएको वा सडकको सञ्चालन अवस्था पुष्टि गर्न सक्दैनन्।",
    ),
    "safe_route": (
        "Mapped reachability does not establish safe or usable travel.",
        "नक्सामा देखिएको पहुँचले यात्रा सुरक्षित वा प्रयोगयोग्य छ भन्ने पुष्टि गर्दैन।",
    ),
    "cause": (
        "The selected run does not establish the event's cause.",
        "छानिएको रनले घटनाको कारण निर्धारण गर्दैन।",
    ),
    "forecast": (
        "These observations are not a forecast of future flooding or travel conditions.",
        "यी अवलोकन भविष्यको बाढी वा यात्राको अवस्थाको पूर्वानुमान होइनन्।",
    ),
    "out_of_run": (
        "This request is outside the authorised run or selected mapped asset.",
        "यो अनुरोध अनुमति दिइएको रन वा छानिएको नक्साङ्कित सम्पत्तिभन्दा बाहिर छ।",
    ),
    "unsupported": (
        "This request cannot be answered by the supported evidence queries. "
        "Choose a predefined question.",
        "यो अनुरोधलाई उपलब्ध प्रमाणसम्बन्धी प्रश्नहरूले उत्तर दिन सक्दैनन्। पूर्वनिर्धारित प्रश्न छान्नुहोस्।",
    ),
    "evidence_unavailable": (
        "The evidence snapshot could not be verified. No impact facts are returned.",
        "प्रमाणको स्न्यापशट प्रमाणित हुन सकेन। प्रभावसम्बन्धी तथ्य दिइएको छैन।",
    ),
    "llm_unavailable": (NOTICE["en"], NOTICE["ne"]),
}


class Refusal(Contract):
    outcome: Literal["refused"] = "refused"
    reason: RefusalReason
    language: Language
    text: str
    native_review: str = NATIVE_REVIEW
    claim_ids: tuple[()] = ()  # Policy text makes no impact claim or fabricated citation.


def refusal(reason: RefusalReason, language: Language) -> Refusal:
    if language not in {"en", "ne"}:
        raise ValueError("Unsupported language")
    return Refusal(
        reason=reason, language=language, text=TEXT[reason][0 if language == "en" else 1]
    )


def screen_question(question: str, language: Language) -> Refusal | None:
    if language not in {"en", "ne"}:
        raise ValueError("Unsupported language")
    if not isinstance(question, str) or not question.strip() or len(question) > 8000:
        return refusal("unsupported", language)
    text = unicodedata.normalize("NFKC", question).casefold()
    text = "".join(c for c in text if unicodedata.category(c) != "Cf")
    for reason, pattern in PATTERNS:
        if re.search(pattern, text):
            return refusal(reason, language)
    if not re.search(SUPPORTED, text):
        return refusal("unsupported", language)
    return None


def guarded_query(
    question: str,
    language: Language,
    context: EvidenceContext,
    authorised_ids: frozenset[str],
    client: ServingClient,
) -> Refusal | QueryResult:
    if context.evidence_id not in authorised_ids:
        return refusal("out_of_run", language)
    blocked = screen_question(question, language)
    if blocked is not None:
        return blocked
    if context.selected_subject is None:
        text = unicodedata.normalize("NFKC", question).casefold()
        text = "".join(c for c in text if unicodedata.category(c) != "Cf")
        if re.search(SELECTED_ASSET, text):
            return refusal("unsupported", language)
    try:
        return select_query(question, context, authorised_ids, client)
    except LLMUnavailable:
        return refusal("llm_unavailable", language)
    except ToolRejected as exc:
        reason: RefusalReason = "unsupported"
        if str(exc) in {"run_not_authorised", "feature_outside_run"}:
            reason = "out_of_run"
        elif str(exc) == "invalid_evidence_context":
            reason = "evidence_unavailable"
        return refusal(reason, language)
