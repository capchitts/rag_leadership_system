import json
import re

from core.schemas import ExecutiveReportResult


def extract_json_text(raw_response: str) -> str:
    if not raw_response:
        raise ValueError("Empty LLM response")

    text = raw_response.strip()
    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No valid JSON object found: {raw_response}")

    return text[start:end + 1]


def parse_executive_report(raw_response: str) -> ExecutiveReportResult:
    json_text = extract_json_text(raw_response)
    parsed = json.loads(json_text)
    return ExecutiveReportResult.model_validate(parsed)