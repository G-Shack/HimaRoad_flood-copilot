"""Authored bilingual report headings; native review is explicitly pending."""

HEADINGS = {
    "title": ("Situation report", "अवस्था प्रतिवेदन"),
    "context": ("Area / event / processing", "क्षेत्र / घटना / प्रशोधन"),
    "acquisitions": ("Satellite acquisitions and offsets", "उपग्रह अवलोकन र समय अन्तर"),
    "method": ("Recorded method and thresholds", "अभिलेखित विधि र सीमाहरू"),
    "coverage": ("Coverage and inclusion sensitivity", "अवलोकनको पहुँच र समावेशको संवेदनशीलता"),
    "hazards": ("Observed hazard areas", "अवलोकन भएका जोखिम क्षेत्र"),
    "impact": ("Potentially affected mapped assets", "सम्भावित प्रभावित नक्साङ्कित सम्पत्ति"),
    "access": ("Hypothetical access scenarios", "काल्पनिक पहुँच परिदृश्य"),
    "limitations": ("Limitations", "सीमाहरू"),
    "unavailable": ("Unavailable / unknown; not zero.", "उपलब्ध छैन / अज्ञात; शून्य होइन।"),
    "appendix": (
        "Complete evidence and source limitations: report-facts.json / "
        "report-claims.csv. Bracketed references resolve to full claim IDs in the "
        "export.",
        "पूर्ण प्रमाण र स्रोतका सीमाहरू: report-facts.json / report-claims.csv। "
        "कोष्ठकका उद्धरणले निर्यातमा पूरा प्रमाण पहिचान जनाउँछन्।",
    ),
    "map": (
        "Run layers only; no travel-condition assessment.",
        "रनका तह मात्र; यात्राको अवस्था मूल्याङ्कन गरिएको छैन।",
    ),
    "legend": (
        "Blue: possible water; ochre: disturbance; grey: unobservable; dark lines: "
        "roads; green points: settlements. Map source claims: map-claims.json.",
        "नीलो: सम्भावित पानी; पहेँलो: परिवर्तन; खरानी: अवलोकन गर्न नसकिने; गाढा "
        "रेखा: सडक; हरियो बिन्दु: बस्ती। नक्साको प्रमाण: map-claims.json।",
    ),
    "synthetic": ("SYNTHETIC DEMONSTRATION", "कृत्रिम प्रदर्शन"),
    "warning": (
        "Satellite times are not event times. Unknown observations never imply zero "
        "impact.",
        "उपग्रहको समय घटनाको समय होइन। "
        "अज्ञात अवलोकनको अर्थ प्रभाव शून्य होइन।",
    ),
    "name_fallback": (
        "Original mapped name (Nepali name unavailable)",
        "मूल नक्साको नाम (नेपाली नाम उपलब्ध छैन)",
    ),
}


def text(key: str, language: str) -> str:
    if language not in {"en", "ne"}:
        raise ValueError("Unsupported language")
    return HEADINGS[key][0 if language == "en" else 1]
