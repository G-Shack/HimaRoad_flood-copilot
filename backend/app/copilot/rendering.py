"""Turn-bound control plans; factual prose is authored and claim-backed only."""

import re
from dataclasses import dataclass
from datetime import date, datetime
from secrets import token_hex
from typing import Annotated, Literal, cast

from pydantic import Field, TypeAdapter, ValidationError

from backend.app.copilot.offline import NATIVE_REVIEW, Language
from backend.app.copilot.serving import LLMUnavailable, ServingClient
from backend.app.copilot.tools import EvidenceContext, ToolRejected, tool_definitions
from backend.app.evidence.claims import verify_claim
from backend.app.evidence.limitations import CATALOGUE
from backend.app.evidence.queries import QueryResult
from backend.app.reports.formatting import event_day, number, timestamp
from backend.app.schemas import SHA256, Answer, Contract, EvidenceClaim


class TextSegment(Contract):
    type: Literal["text"]
    text: Literal["Evidence follows.", "प्रमाण तल दिइएको छ।"]


class ReferenceSegment(Contract):
    type: Literal["claim", "feature", "date"]
    claim_id: SHA256


Segment = Annotated[TextSegment | ReferenceSegment, Field(discriminator="type")]


class SegmentPlan(Contract):
    mode: Literal["segments"]
    turn_id: str
    segments: list[Segment] = Field(max_length=256)


class ActionPlan(Contract):
    mode: Literal["action"]
    turn_id: str
    template_id: Literal["evidence"]
    claim_ids: list[SHA256] = Field(max_length=255)


Plan = Annotated[SegmentPlan | ActionPlan, Field(discriminator="mode")]
PLAN = TypeAdapter(Plan)


@dataclass(frozen=True)
class RenderTurn:
    """Internal trusted object. Never deserialize a turn from an API request."""

    id: str
    result: QueryResult
    claims: tuple[EvidenceClaim, ...]


def prepare_turn(
    result: QueryResult, context: EvidenceContext, authorised_ids: frozenset[str]
) -> RenderTurn:
    if context.evidence_id not in authorised_ids:
        raise ToolRejected("run_not_authorised")
    tool_definitions(context)  # Recheck snapshot hashes/membership, including nested values.
    known = {claim.id: claim for claim in context.claims}
    supplied = (*result.claims, *result.coverage, *result.limitations)
    for claim in supplied:
        verify_claim(claim)
    selected = {c.id: c for c in supplied}
    if {c.id for c in result.limitations} != {
        c.id for c in context.claims if c.kind == "limitation"
    }:
        raise ToolRejected("invalid_evidence_context")
    for identifier, claim in selected.items():
        verify_claim(claim)
        if identifier not in known or claim != known[identifier]:
            raise ToolRejected("invalid_evidence_context")
    # Coverage and limitations cannot be dropped by an intermediate consumer.
    if {c.id for c in result.coverage} != {
        c.id for c in context.claims if c.kind == "sensitivity"
    } and not any(
        c.kind == "evidence_validation" and c.value["result"] == "failed" for c in context.claims
    ):
        raise ToolRejected("invalid_evidence_context")
    # QueryResult must originate in deterministic query dispatch, never model input.
    integrity = next(c for c in context.claims if c.kind == "evidence_validation")
    if integrity.value["result"] == "failed" and (
        result.availability != "unavailable" or result.claims != (integrity,) or result.coverage
    ):
        raise ToolRejected("invalid_evidence_context")
    return RenderTurn(token_hex(16), result, tuple(sorted(selected.values(), key=lambda c: c.id)))


def reference_type(claim: EvidenceClaim) -> Literal["claim", "feature", "date"]:
    if claim.kind == "feature":
        return "feature"
    if claim.kind in {"event_date", "observation_timestamp"}:
        return "date"
    return "claim"


def control_schema(turn: RenderTurn, mode: Literal["segments", "action"]) -> dict:
    """Only opaque IDs and syntactic kinds reach the model, never source text/values."""
    ids = list(control_references(turn))
    properties: dict = {
        "mode": {"const": mode, "enum": [mode], "type": "string"},
        "turn_id": {"const": turn.id, "enum": [turn.id], "type": "string"},
    }
    if mode == "action":
        properties |= {
            "template_id": {"const": "evidence", "enum": ["evidence"], "type": "string"},
            "claim_ids": {"type": "array", "items": {"enum": ids}, "maxItems": 255},
        }
    else:
        properties["segments"] = {
            "type": "array",
            "maxItems": 256,
            "items": {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {
                            "type": {
                                "const": reference_type(c),
                                "enum": [reference_type(c)],
                                "type": "string",
                            },
                            "claim_id": {"const": ref, "enum": [ref], "type": "string"},
                        },
                        "required": ["type", "claim_id"],
                        "additionalProperties": False,
                    }
                    for ref, c in zip(ids, turn.claims, strict=True)
                ]
                + [
                    {
                        "type": "object",
                        "properties": {
                            "type": {"const": "text", "enum": ["text"], "type": "string"},
                            "text": {"enum": ["Evidence follows.", "प्रमाण तल दिइएको छ।"]},
                        },
                        "required": ["type", "text"],
                        "additionalProperties": False,
                    }
                ]
            },
        }
    return {
        "type": "function",
        "function": {
            "name": "render_evidence",
            "description": "Retain every supplied evidence reference once. Do not write facts.",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


def control_references(turn: RenderTurn) -> dict[str, str]:
    """Short tokens have meaning only alongside this trusted turn's fresh nonce."""
    return {f"r{i:x}": claim.id for i, claim in enumerate(turn.claims)}


def resolve_control(turn: RenderTurn, payload: dict) -> dict:
    """Resolve wire tokens without coercion; strict full-hash validation follows."""
    if payload.get("turn_id") != turn.id:
        raise ValueError("wrong rendering turn")
    known = control_references(turn)

    def resolve(value: object) -> str:
        if not isinstance(value, str) or value not in known:
            raise ValueError("foreign rendering reference")
        return known[value]

    expanded = dict(payload)
    if payload.get("mode") == "action":
        values = payload.get("claim_ids")
        if not isinstance(values, list) or len(values) > 255:
            raise ValueError("invalid rendering references")
        expanded["claim_ids"] = [resolve(value) for value in values]
    elif payload.get("mode") == "segments":
        segments = payload.get("segments")
        if not isinstance(segments, list) or len(segments) > 256:
            raise ValueError("invalid rendering segments")
        expanded["segments"] = []
        for item in segments:
            if not isinstance(item, dict):
                raise ValueError("invalid rendering segment")
            segment = dict(item)
            if "claim_id" in segment:
                segment["claim_id"] = resolve(segment["claim_id"])
            expanded["segments"].append(segment)
    else:
        raise ValueError("invalid rendering mode")
    return expanded


# Initial core templates. Detailed report prose and native review belong to Phase04.
LABELS = {
    "corroborated_road_cut": (
        "Modelled disconnections with corroborated evidence",
        "थप प्रमाण भएका मोडेलका सम्पर्क विच्छेद",
    ),
    "depends_only_on_unknown": (
        "Mapped connections depending on unknown links",
        "अज्ञात खण्डमा निर्भर नक्साको सम्पर्क",
    ),
    "settlements_total": ("Mapped settlements", "नक्सामा रहेका बस्तीहरू"),
    "settlements_from_building_clusters": (
        "Settlements represented by building clusters",
        "भवन समूहले प्रतिनिधित्व गरेका बस्तीहरू",
    ),
    "length_m": ("Path length (m)", "मार्ग लम्बाइ (मि.)"),
    "corridor_m": ("Corridor radius (m)", "मार्ग क्षेत्र त्रिज्या (मि.)"),
    "cells": ("DEM cells", "DEM कोषहरू"),
    "potentially_road_cut": ("Modelled disconnected settlements", "मोडेलमा सम्पर्क टुटेका बस्तीहरू"),
    "access_uncertain": ("Settlements with uncertain access", "पहुँच अनिश्चित भएका बस्तीहरू"),
    "trail_access_possible": ("Possible mapped trail access", "नक्सामा सम्भावित गोरेटो पहुँच"),
    "primary_care_potentially_cut": (
        "Potential primary-care disconnections",
        "स्वास्थ्य केन्द्रसँग सम्भावित सम्पर्क विच्छेद",
    ),
    "potentially_exposed_buildings": (
        "Potentially exposed buildings",
        "सम्भावित जोखिममा रहेका भवनहरू",
    ),
    "suspected_affected_links": ("Potentially affected links", "सम्भावित प्रभावित सडक खण्डहरू"),
    "valid_post_observation_percent": (
        "Post-observation coverage (%)",
        "पछिल्लो अवलोकनको समेटिएको भाग (%)",
    ),
    "valid_change_observation_percent": (
        "Change-observation coverage (%)",
        "परिवर्तन अवलोकनको समेटिएको भाग (%)",
    ),
    "flood_extent": ("Possible water area", "सम्भावित पानीको क्षेत्रफल"),
    "disturbance": ("Possible disturbance area", "सम्भावित परिवर्तनको क्षेत्रफल"),
    "possible_debris": ("possible debris (unvalidated)", "सम्भावित गेग्रान (प्रमाणित नभएको)"),
    "optical_water": ("Optical possible water area", "अप्टिकल सम्भावित पानी क्षेत्रफल"),
    "optical_disturbance": ("Optical possible disturbance area", "अप्टिकल सम्भावित परिवर्तन क्षेत्रफल"),
    "unobservable": ("Unobservable area", "अवलोकन गर्न नसकिने क्षेत्रफल"),
    "area_m2": ("Area (m²)", "क्षेत्रफल (m²)"),
    "observed_area_m2": ("Observed coverage (m²)", "अवलोकनले समेटेको क्षेत्रफल (m²)"),
    "water_only": ("Water-only inclusion", "पानी मात्र समावेश गर्दा"),
    "water_or_disturbance": ("Water or disturbance inclusion", "पानी वा परिवर्तन समावेश गर्दा"),
    "unobservable_features": ("Unobservable mapped features", "अवलोकन गर्न नसकिने नक्साका वस्तुहरू"),
    "partial_coverage_features": ("Partially observed features", "आंशिक अवलोकन भएका वस्तुहरू"),
    "feature_count": ("Mapped features", "नक्साका वस्तुहरू"),
    "observed_change_links": ("Observed-change links", "परिवर्तन अवलोकन भएका सडक खण्डहरू"),
    "corroborated_links": ("Corroborated links", "थप प्रमाण भएका सडक खण्डहरू"),
}
LIMITS_NE = {
    "unet_himalayan_candidate": (
        "हिमाली उपत्यकामा UNetको पानी प्रमाणित नभएको सम्भावित सङ्केत हो। "
        "अभिलेखित मोडेल पहिचान र विकास चरणको क्यालिब्रेसनले अन्य भूभागमा क्षमता पुष्टि गर्दैन।"
    ),
    "hazard3_observability": (
        "अभिलेखित layover/shadow मास्क अनुमानित हुन्; अवलोकन नभएका क्षेत्र अज्ञात रहन्छन्। "
        "धेरै उपग्रह मितिका तह जोडिएका छन्; वस्तुको समय परिवर्तन देखाउने पहिलो अवलोकन हो।"
    ),
    "hazard3_optical": (
        "अप्टिकल प्रमाण मात्र अत्यन्त प्रतिकूल परिदृश्यमा पर्छ र लामो अन्तरका सङ्केत दबाइन्छन्। "
        "अभिलेखित NDBI सीमा प्रयोग हुँदैन; अप्टिकल समर्थनले सम्भावित गेग्रान प्रमाणित गर्दैन।"
    ),
    "flow_not_forecast": "DEM बहाव मार्ग भूभागको रेखाङ्कन हो; डुबान पूर्वानुमान वा यात्राको सिफारिस होइन।",
    "mapped_network": "नक्साका बस्ती, सडक र गन्तव्य अपूर्ण हुन सक्छन्; नक्साबाहिरका वस्तुहरू समावेश छैनन्।",
    "hypothetical": "परिदृश्यका अवरोध र सम्पर्क विच्छेद काल्पनिक हुन्; पुष्टि भएका अवस्था होइनन्।",
    "acquisition_boundary": (
        "उपग्रह अवलोकनको समय घटनाको समय होइन; अन्तर दिइएको घटना मितिको सीमाबाट गणना हुन्छ।"
    ),
    "resolution": "पत्ता लगाउने, जोखिम र सडकसँग जोड्ने नतिजा अभिलेखित रिजोल्युसन र सीमामा निर्भर छन्।",
    "synthetic": "कृत्रिम प्रदर्शन; वास्तविक घटनाको प्रमाण होइन।",
    "synthetic_components": (
        "प्रशोधनका केही भाग कृत्रिम भनेर चिनाइएका छन्। नतिजा प्रदर्शनका लागि हुन्; "
        "वास्तविक घटनाको प्रभावको प्रमाण होइनन्।"
    ),
    "model_evaluation": ("अभिलेखित मोडेलको पहिचान र सञ्चालन बिन्दुले स्वतन्त्र शुद्धता वा प्रभाव पुष्टि गर्दैनन्।"),
    "producer_unknown": "उत्पादक कमिट अज्ञात छ; एडेप्टर पिन उत्पादकको पहिचान होइन।",
    "incomplete": "रन अपूर्ण छ; प्रभावका अवलोकन र सङ्ख्या उपलब्ध छैनन्।",
    "optical_absent": "घटनापछिको अप्टिकल अवलोकन अभिलेखित छैन; अप्टिकल पुष्टिकरण उपलब्ध छैन।",
    "zero_scenario2_without_corroboration": (
        "परिदृश्य 2 मा मोडेलले सम्पर्क विच्छेद शून्य देखाउँछ र "
        "राडार/अप्टिकल पुष्टिकरण भएका खण्ड छैनन्; परिवर्तन देखिएका खण्ड बाँकी छन्। "
        "यसले प्रभावित बस्ती नभएको पुष्टि गर्दैन।"
    ),
    "terrain_observability": "समीक्षा गरिएको करारले राडार छाया वा लेओभरको समेटिएको क्षेत्र स्थापित गर्दैन।",
    "nonallowed_source": "अभिलेखित स्रोतको नियमअनुसार समीक्षा आवश्यक छ; यो रन नियमअनुकूल भनेर स्वीकृत छैन।",
}


def _word(en: str, ne: str, language: Language) -> str:
    return en if language == "en" else ne


def _number(value: object, language: Language) -> str:
    if value is None:
        return _word("unknown", "अज्ञात", language)
    if type(value) not in {int, float}:
        raise ValueError("invalid numeric claim projection")
    return number(cast(int | float, value), language)


def _fields(value: dict, fields: tuple[str, ...], language: Language) -> str:
    index = 0 if language == "en" else 1
    return "; ".join(
        f"{LABELS[key][index]}: {_number(value[key], language)}" for key in fields if key in value
    )


def _identifier(value: object, language: Language) -> str:
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_:/.-]{1,256}", value):
        return value
    return _word("unknown identifier", "पहिचान अज्ञात", language)


def _access(value: object, language: Language) -> str:
    if value is None:
        return _word("unknown", "अज्ञात", language)
    if type(value) is not bool:
        raise ValueError("invalid mapped-access projection")
    return (
        _word("mapped connection", "नक्सामा सम्पर्क", language)
        if value
        else _word("no mapped connection", "नक्सामा सम्पर्क छैन", language)
    )


def _source_card(claim: EvidenceClaim, language: Language) -> str | None:
    parts = claim.label.strip("/").split("/")
    root, field = parts[0], parts[-1]
    scenes = {
        "pre": ("Radar pre acquisition", "घटनाअघिको राडार अवलोकन"),
        "post": ("Radar post acquisition", "घटनापछिको राडार अवलोकन"),
        "optical_pre": ("Optical pre acquisition", "घटनाअघिको अप्टिकल अवलोकन"),
        "optical_post": ("Optical post acquisition", "घटनापछिको अप्टिकल अवलोकन"),
    }
    index = 0 if language == "en" else 1
    if root in scenes and field == "timestamp":
        if claim.value is None:
            value = _word("unknown", "अज्ञात", language)
        else:
            datetime.fromisoformat(claim.value.replace("Z", "+00:00"))
            value = timestamp(claim.value, language)
        return (
            scenes[root][index]
            + ": "
            + value
            + _word(
                "; satellite time, not event time.", "; उपग्रहको समय, घटनाको समय होइन।", language
            )
        )
    if root in {*scenes, "dem", "hand"} and field == "source":
        known_sources = {
            "SYNTHETIC TEST ONLY",
            "SYNTHETIC OPTICAL",
            "COPERNICUS/S1_GRD",
            "COPERNICUS/S2_SR_HARMONIZED",
            "COPERNICUS/DEM/GLO30",
            "COPERNICUS_30",
        }
        if isinstance(claim.value, str) and claim.value in known_sources:
            return _word("Recorded input source: ", "अभिलेखित आगत स्रोत: ", language) + claim.value
    if root in scenes and field in {"scene_id", "platform"}:
        return (
            scenes[root][index]
            + _word(" recorded identifier: ", " अभिलेखित पहिचान: ", language)
            + _identifier(claim.value, language)
        )
    if root in {"osm", "cutoff"} and field in {"snapshot", "max_edit_timestamp", "cutoff"}:
        if claim.value is None:
            value = _word("unknown", "अज्ञात", language)
        else:
            datetime.fromisoformat(claim.value.replace("Z", "+00:00"))
            value = claim.value
        label = {
            "snapshot": ("Recorded OSM snapshot", "अभिलेखित OSM स्न्यापसट"),
            "max_edit_timestamp": ("Recorded latest OSM edit", "अभिलेखित अन्तिम OSM सम्पादन"),
            "cutoff": ("Recorded OSM cutoff", "अभिलेखित OSM समय सीमा"),
        }
        return label[field][index] + ": " + value
    if root == "provider" and claim.value in {"local", "cdse", "gee"}:
        return _word("Recorded provider: ", "अभिलेखित प्रदायक: ", language) + claim.value
    if root == "temporal_gaps_hours":
        labels = {
            "post_after_event": (
                "Post acquisition offset (hours)",
                "पछिल्लो अवलोकनको समय अन्तर (घण्टा)",
            ),
            "latest_pre_before_event": (
                "Latest pre acquisition offset (hours)",
                "अघिल्लो अन्तिम अवलोकनको समय अन्तर (घण्टा)",
            ),
        }
        if field in labels:
            return (
                labels[field][index]
                + ": "
                + _number(claim.value, language)
                + _word(
                    "; measured from the supplied event-date boundary, not event time.",
                    "; दिइएको घटना मितिको सीमाबाट गणना, घटनाको समयबाट होइन।",
                    language,
                )
            )
    return None


def _route_card(claim: EvidenceClaim, language: Language) -> str:
    value = claim.value
    outcomes = {
        "found": ("Mapped route found", "नक्सामा मार्ग भेटियो"),
        "none": ("No route found in the modelled network", "मोडेलको नक्साको सञ्जालमा मार्ग भेटिएन"),
        "unassessable": ("Mapped route cannot be assessed", "नक्साको मार्गको मूल्याङ्कन गर्न सकिँदैन"),
    }
    index = 0 if language == "en" else 1
    result = outcomes[value["result"]][index]
    removed = ", ".join(_identifier(link, language) for link in value["removed_link_ids"])
    if not removed:
        removed = _word("no recorded removals", "अभिलेखित हटाइएका खण्ड छैनन्", language)
    return (
        result
        + _word("; recorded mode: ", "; अभिलेखित माध्यम: ", language)
        + _identifier(claim.label, language)
        + _word("; selected settlement: ", "; छानिएको बस्ती: ", language)
        + _identifier(value["settlement_id"], language)
        + _word("; mapped target: ", "; नक्साको गन्तव्य: ", language)
        + _identifier(value["target_id"], language)
        + _word("; network cost (m): ", "; सञ्जाल लागत (मि.): ", language)
        + _number(value["cost_m"], language)
        + _word("; scenario-removed link IDs: ", "; परिदृश्यले हटाएका खण्डका पहिचान: ", language)
        + removed
        + _word(
            "; mapped reachability does not establish travel conditions.",
            "; नक्साको पहुँचले यात्राको अवस्था पुष्टि गर्दैन।",
            language,
        )
    )


def _repair_card(claim: EvidenceClaim, language: Language) -> str:
    value = claim.value
    if value["availability"] == "unavailable":
        return _word(
            "Single-link reopening gains unavailable; this does not mean no repair benefit.",
            "एक खण्ड खोल्दाको लाभ उपलब्ध छैन; यसले मर्मतको लाभ नभएको पुष्टि गर्दैन।",
            language,
        )
    if value["availability"] != "available" or not isinstance(value["rankings"], list):
        raise ValueError("invalid repair projection")
    rows = [
        _word("Link ID: ", "खण्डको पहिचान: ", language)
        + _identifier(row["link_id"], language)
        + _word("; reconnected mapped settlements: ", "; पुनः जोडिएका नक्साका बस्ती: ", language)
        + _number(row["settlement_count"], language)
        + _word(
            "; potentially exposed buildings assigned to those settlements: ",
            "; ती बस्तीमा जोडिएका सम्भावित जोखिमका भवन: ",
            language,
        )
        + _number(row["exposed_buildings_assigned_to_reconnected_settlements"], language)
        for row in value["rankings"]
    ]
    if not rows:
        rows = [
            _word("No ranked single-link candidates.", "क्रमबद्ध एकल खण्डका उम्मेदवार छैनन्।", language)
        ]
    return (
        _word(
            "Hypothetical single-link reopening gains: ", "काल्पनिक एकल खण्ड खोल्दाको लाभ: ", language
        )
        + " | ".join(rows)
        + _word(
            " Joint repairs may help even when individual gains are zero. "
            "Buildings use nearest-settlement assignment; affiliation is unknown. "
            "Not a verified repair or travel recommendation.",
            " एकल लाभ शून्य भए पनि संयुक्त मर्मतले मद्दत गर्न सक्छ। "
            "भवन नजिकको बस्तीसँग जोडिन्छन्; वास्तविक सम्बन्ध अज्ञात छ। "
            "यो पुष्टि भएको मर्मत वा यात्राको सिफारिस होइन।",
            language,
        )
    )


def _card(claim: EvidenceClaim, language: Language) -> str:
    index = 0 if language == "en" else 1
    text = _word(
        "Detailed presentation unavailable; inspect the cited evidence record.",
        "विस्तृत प्रस्तुति उपलब्ध छैन; उद्धृत प्रमाण अभिलेख हेर्नुहोस्।",
        language,
    )
    if claim.kind == "source":
        text = _source_card(claim, language) or text
    elif claim.kind == "feature":
        text = _word("Mapped feature ID: ", "नक्साको वस्तुको पहिचान: ", language) + _identifier(
            claim.subject, language
        )
    elif claim.kind == "repair_priority":
        text = _repair_card(claim, language)
    elif claim.kind == "primary_care_access":
        text = (
            _word(
                "Mapped primary-care reachability, not service availability; settlement ID: ",
                "नक्सामा स्वास्थ्य केन्द्रको पहुँच; सेवाको उपलब्धता होइन; बस्तीको पहिचान: ",
                language,
            )
            + _identifier(claim.subject, language)
            + "; "
            + "; ".join(
                (
                    _word("baseline", "आधार अवस्था", language)
                    if scenario == "baseline"
                    else _word("scenario ", "परिदृश्य ", language) + scenario
                )
                + ": "
                + _access(claim.value[key], language)
                for scenario, key in (
                    ("baseline", "primary_care_baseline_accessible"),
                    ("1", "primary_care_scenario1_accessible"),
                    ("2", "primary_care_scenario2_accessible"),
                    ("3", "primary_care_scenario3_accessible"),
                )
            )
        )
    elif claim.kind == "summary" and claim.label.rsplit("/", 1)[-1] in LABELS:
        key = claim.label.rsplit("/", 1)[-1]
        text = f"{LABELS[key][index]}: {_number(claim.value, language)}"
    elif claim.kind == "area" and claim.label in LABELS:
        text = (
            LABELS[claim.label][index]
            + ": "
            + _fields(claim.value, ("area_m2", "observed_area_m2"), language)
        )
    elif claim.kind == "sensitivity":
        text = _word(
            "Inclusion sensitivity, not a confidence interval. ",
            "समावेशको संवेदनशीलता; विश्वास अन्तराल होइन। ",
            language,
        ) + _fields(
            claim.value,
            (
                "water_only",
                "water_or_disturbance",
                "unobservable_features",
                "partial_coverage_features",
                "feature_count",
            ),
            language,
        )
    elif claim.kind == "limitation":
        key = claim.label
        text = (
            (CATALOGUE[key] if language == "en" else LIMITS_NE[key])
            if key in CATALOGUE
            else _word(
                "Additional source limitation recorded; inspect the citation.",
                "स्रोतको थप सीमा अभिलेखित छ; उद्धरण हेर्नुहोस्।",
                language,
            )
        )
        if key == "zero_scenario2_without_corroboration":
            text += " " + _fields(
                claim.value["details"], ("observed_change_links", "corroborated_links"), language
            )
    elif claim.kind == "event_date":
        parsed = date.fromisoformat(claim.value)
        text = _word("Supplied event date: ", "दिइएको घटना मिति: ", language) + event_day(
            parsed.isoformat(), language
        )
    elif claim.kind == "observation_timestamp":
        parsed_time = datetime.fromisoformat(claim.value.replace("Z", "+00:00"))
        text = _word(
            "Satellite acquisition time, not event time: ",
            "उपग्रह अवलोकनको समय, घटनाको समय होइन: ",
            language,
        ) + timestamp(parsed_time.isoformat(), language)
    elif claim.kind == "evidence_validation":
        result = claim.value["result"]
        if result in {"passed", "failed", "unavailable"}:
            text = (
                _word(
                    "Evidence integrity (not mapping accuracy): ",
                    "प्रमाण अखण्डता (नक्साको शुद्धता होइन): ",
                    language,
                )
                + result
            )
    elif claim.kind == "status" and claim.value in {"complete", "incomplete"}:
        text = _word("Run processing status: ", "रन प्रशोधनको अवस्था: ", language) + claim.value
    elif claim.kind == "flow_path" and claim.value["availability"] == "available":
        text = _word(
            "DEM-derived flow path; not an inundation forecast. ",
            "DEMबाट बहाव मार्ग; डुबान पूर्वानुमान होइन। ",
            language,
        )
        text += _word("Recorded ID: ", "अभिलेखित पहिचान: ", language) + _identifier(
            claim.subject, language
        )
        if claim.value.get("kind") == "flow_path":
            text += "; " + _fields(claim.value, ("length_m", "corridor_m", "cells"), language)
    elif claim.kind == "flow_path" and claim.value["availability"] == "unavailable":
        text = _word(
            "Flow path unavailable in this HimaRoad run.",
            "यो HimaRoad रनमा बहाव मार्ग उपलब्ध छैन।",
            language,
        )
    if claim.kind in {"feature", "sensitivity"} and claim.label in {
        "roads",
        "buildings",
        "settlements",
    }:
        layer_names = {
            "roads": ("Roads", "सडकहरू"),
            "buildings": ("Buildings", "भवनहरू"),
            "settlements": ("Settlements", "बस्तीहरू"),
        }
        text = layer_names[claim.label][index] + ": " + text
    if claim.kind == "route":
        text = _route_card(claim, language)
    # Exact scenario/status tokens are evidence values, never model prose.
    if claim.scenario in {"baseline", "1", "2", "3"}:
        text += (
            " " + _word("Hypothetical scenario: ", "काल्पनिक परिदृश्य: ", language) + claim.scenario
        )
    if claim.kind in {"feature", "route", "primary_care_access"}:
        text += " " + _word("Evidence status: ", "प्रमाणको अवस्था: ", language) + claim.status.value
    return f"{text} [{claim.id}]"


def render_template(turn: RenderTurn, language: Language = "en") -> Answer:
    if language not in {"en", "ne"}:
        raise ValueError("Unsupported language")
    for claim in turn.claims:
        verify_claim(claim)
    heading = _word("Selected evidence records follow.", "छानिएका प्रमाण अभिलेख तल छन्।", language)
    if turn.result.availability == "unavailable":
        heading = _word(
            "Requested evidence unavailable; this does not mean zero impact.",
            "मागिएको प्रमाण उपलब्ध छैन; यसको अर्थ प्रभाव शून्य होइन।",
            language,
        )
    elif not turn.result.claims:
        heading = _word(
            "No matching records in this query; this does not establish absence of impact.",
            "यो प्रश्नमा मिल्ने अभिलेख छैनन्; यसले प्रभाव नभएको पुष्टि गर्दैन।",
            language,
        )
    failed = any(
        c.kind == "evidence_validation" and c.value["result"] == "failed"
        for c in turn.result.claims
    )
    if failed:
        heading = _word(
            "Evidence integrity failed; impact observations and counts are withheld.",
            "प्रमाण अखण्डता जाँच असफल भयो; प्रभावका अवलोकन र सङ्ख्या रोकिएका छन्।",
            language,
        )
    withheld = _word(
        "Limitation details withheld because evidence integrity failed; inspect the citation.",
        "प्रमाण अखण्डता जाँच असफल भएकाले सीमाको विवरण रोकिएको छ; उद्धरण हेर्नुहोस्।",
        language,
    )
    cards = tuple(
        f"{withheld} [{c.id}]"
        if failed and c.kind == "limitation" and c.label != "educational"
        else _card(c, language)
        for c in turn.claims
    )
    return Answer(
        language=language,
        segments=(heading, *cards, *((NATIVE_REVIEW,) if language == "ne" else ())),
        claim_ids=tuple(c.id for c in turn.claims),
        feature_refs=tuple(c.subject for c in turn.claims if c.kind == "feature"),
        limitations=tuple(c.id for c in turn.claims if c.kind == "limitation"),
        rendered_by="template",
        validation_result="passed",
    )


def render_plan(turn: RenderTurn, payload: dict, language: Language = "en") -> Answer:
    plan = PLAN.validate_python(payload, strict=True)
    if plan.turn_id != turn.id:
        raise ValueError("wrong rendering turn")
    if isinstance(plan, ActionPlan):
        identifiers = plan.claim_ids
    else:
        refs = [s for s in plan.segments if isinstance(s, ReferenceSegment)]
        texts = [s for s in plan.segments if isinstance(s, TextSegment)]
        if len(texts) > 1 or any(
            s.text != _word("Evidence follows.", "प्रमाण तल दिइएको छ।", language) for s in texts
        ):
            raise ValueError("invalid connective text")
        known = {c.id: reference_type(c) for c in turn.claims}
        if any(known.get(s.claim_id) != s.type for s in refs):
            raise ValueError("wrong placeholder kind or scope")
        identifiers = [s.claim_id for s in refs]
    if len(identifiers) != len(set(identifiers)) or set(identifiers) != {c.id for c in turn.claims}:
        raise ValueError("missing, duplicated or foreign evidence")
    # Canonical order prevents a model from burying or rearranging mandatory caveats.
    return render_template(turn, language).model_copy(update={"rendered_by": "llm"})


def validate_answer(turn: RenderTurn, answer: Answer) -> bool:
    expected = render_template(turn, answer.language)
    return answer.rendered_by in {"llm", "template"} and answer.model_dump(
        exclude={"rendered_by"}
    ) == expected.model_dump(exclude={"rendered_by"})


def render_or_fallback(turn: RenderTurn, payload: dict, language: Language = "en") -> Answer:
    try:
        answer = render_plan(turn, payload, language)
    except (ValidationError, ValueError, TypeError, KeyError):
        return render_template(turn, language)
    return answer if validate_answer(turn, answer) else render_template(turn, language)


def select_rendering(
    turn: RenderTurn,
    client: ServingClient,
    language: Language = "en",
    mode: Literal["segments", "action"] = "action",
) -> Answer:
    # Large results remain fully cited in template mode; no silent model truncation.
    if len(turn.claims) > 64:
        return render_template(turn, language)
    try:
        proposal = client.propose_tool(
            "Select the supplied evidence rendering plan.", [control_schema(turn, mode)]
        )
    except LLMUnavailable:
        return render_template(turn, language)
    if proposal.name != "render_evidence" or proposal.arguments.get("mode") != mode:
        return render_template(turn, language)
    try:
        payload = resolve_control(turn, proposal.arguments)
    except (ValueError, TypeError, KeyError):
        return render_template(turn, language)
    return render_or_fallback(turn, payload, language)
