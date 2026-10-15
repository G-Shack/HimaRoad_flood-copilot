"""Predefined read-only questions remain usable without a model server."""

from typing import Literal

from backend.app.evidence import queries
from backend.app.reports.native_review import review_notice
from backend.app.schemas import Contract, EvidenceClaim

Language = Literal["en", "ne"]
QuestionID = Literal[
    "summary", "roads", "buildings", "isolation", "sources", "repair", "flow", "integrity"
]
NATIVE_REVIEW = review_notice()
QUESTIONS: dict[QuestionID, tuple[str, str]] = {
    "summary": ("Summarise this run.", "यो रनको सारांश दिनुहोस्।"),
    "roads": ("Which roads are potentially affected?", "कुन सडकहरू सम्भावित रूपमा प्रभावित छन्?"),
    "buildings": (
        "Which buildings are potentially exposed?",
        "कुन भवनहरू सम्भावित रूपमा जोखिममा छन्?",
    ),
    "isolation": (
        "Which settlements are modelled as disconnected?",
        "कुन बस्तीहरूको सम्पर्क टुटेको हुन सक्ने मोडेलले देखाउँछ?",
    ),
    "sources": ("Show sources and limitations.", "स्रोत र सीमाहरू देखाउनुहोस्।"),
    "repair": ("Show hypothetical repair priorities.", "काल्पनिक मर्मत प्राथमिकताहरू देखाउनुहोस्।"),
    "flow": ("Is downstream flow-path evidence available?", "बहाव मार्गको प्रमाण उपलब्ध छ?"),
    "integrity": ("Show evidence integrity checks.", "प्रमाण अखण्डताको जाँच देखाउनुहोस्।"),
}
NOTICE = {
    "en": "LLM unavailable. Predefined evidence questions remain available.",
    "ne": "भाषा मोडेल उपलब्ध छैन। पूर्वनिर्धारित प्रमाणसम्बन्धी प्रश्नहरू उपलब्ध छन्।",
}


class PredefinedQuestion(Contract):
    id: QuestionID
    label: str


class OfflineMenu(Contract):
    mode: Literal["llm_unavailable"] = "llm_unavailable"
    notice: str
    questions: tuple[PredefinedQuestion, ...]
    native_review: str = NATIVE_REVIEW


def offline_menu(language: Language = "en") -> OfflineMenu:
    if language not in {"en", "ne"}:
        raise ValueError("Unsupported language")
    index = 0 if language == "en" else 1
    return OfflineMenu(
        notice=NOTICE[language],
        questions=tuple(
            PredefinedQuestion(id=key, label=value[index]) for key, value in QUESTIONS.items()
        ),
    )


def answer_predefined(
    question: QuestionID, claims: tuple[EvidenceClaim, ...]
) -> queries.QueryResult:
    """Claims must already belong to the caller's authorised evidence snapshot."""
    if question == "roads":
        return queries.list_affected_assets(claims, "roads")
    if question == "buildings":
        return queries.list_affected_assets(claims, "buildings")
    functions = {
        "summary": queries.summarize_run,
        "isolation": queries.list_potentially_isolated_settlements,
        "sources": queries.get_sources_and_limitations,
        "repair": queries.get_repair_priority,
        "flow": queries.trace_flow_path,
        "integrity": queries.get_evidence_validation,
    }
    if question not in functions:
        raise ValueError("Unknown predefined evidence question")
    return functions[question](claims)
