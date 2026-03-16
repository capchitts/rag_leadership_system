import re

SECTION_PATTERNS = {
    "executive_overview": [
        r"executive overview",
        r"overview",
    ],
    "performance_summary": [
        r"performance summary",
        r"financial highlights",
        r"key metrics",
    ],
    "segment_commentary": [
        r"segment commentary",
        r"digital commerce",
        r"physical stores",
        r"private label",
    ],
    "strategic_risks_opportunities": [
        r"strategic risks and opportunities",
        r"risks and opportunities",
        r"strategic risks",
    ],
    "leadership_questions": [
        r"leadership questions",
        r"questions for review",
    ]
}


def detect_section(text: str) -> str:
    lower_text = text.lower()

    for section, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, lower_text):
                return section

    return "general"