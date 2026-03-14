import re
import json
from prompts.faithfulness_prompt import build_faithfulness_prompt
from core.schemas import FaithfulnessResult


def _extract_json(raw_response: str):
    """
    Extract JSON from LLM response.
    Handles markdown code blocks and stray text.
    """
    if not raw_response:
        raise ValueError("Empty LLM response")

    text = raw_response.strip()

    # remove markdown fences
    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)

    # extract first json object
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object found")

    return text[start:end + 1]


def evaluate_faithfulness(llm, answer: str, context: str) -> FaithfulnessResult:
    prompt = build_faithfulness_prompt(answer=answer, context=context)
    raw_response = llm.invoke(prompt)

    try:
        json_text = _extract_json(raw_response)
        parsed = json.loads(json_text)

        return FaithfulnessResult.model_validate(parsed)

    except Exception:
        return FaithfulnessResult(
            faithfulness_score=None,
            directly_supported_claims=[],
            partially_supported_claims=[],
            unsupported_claims=[],
            summary=f"Failed to parse JSON response: {raw_response}",
        )