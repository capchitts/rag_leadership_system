import re
import json
from prompts.groundedness_prompt import build_groundedness_prompt
from core.schemas import GroundednessResult


def _extract_json(raw_response: str):
    """
    Extract JSON from LLM response.
    Handles markdown code blocks and stray text.
    """
    if not raw_response:
        raise ValueError("Empty LLM response")

    text = raw_response.strip()

    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object found")

    return text[start:end + 1]


def evaluate_groundedness(llm, answer: str, context: str) -> GroundednessResult:
    prompt = build_groundedness_prompt(answer=answer, context=context)
    raw_response = llm.invoke(prompt)

    try:
        json_text = _extract_json(raw_response)
        parsed = json.loads(json_text)

        return GroundednessResult.model_validate(parsed)

    except Exception:
        return GroundednessResult(
            groundedness_score=None,
            explanation=f"Failed to parse JSON response: {raw_response}",
            unsupported_statements=[],
        )