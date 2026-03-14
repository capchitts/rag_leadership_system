import re

SECTIONS = [
    "executive summary",
    "risk analysis",
    "financial performance",
    "operations",
    "strategy",
    "market outlook"
]

def detect_section(text):

    t = text.lower()

    for s in SECTIONS:

        if re.search(s, t):
            return s

    return "general"