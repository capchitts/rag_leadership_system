import re
import json

from prompts.groundedness_prompt import build_groundedness_prompt
from core.schemas import GroundednessResult
from utils.logger import get_logger

logger = get_logger(__name__)


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
    logger.info(
        "Groundedness evaluation started",
        extra={"extra_data": {"answer_chars": len(answer), "context_chars": len(context)}},
    )

    prompt = build_groundedness_prompt(answer=answer, context=context)
    raw_response = llm.invoke(prompt)

    try:
        json_text = _extract_json(raw_response)
        parsed = json.loads(json_text)

        result = GroundednessResult.model_validate(parsed)

        logger.info(
            "Groundedness evaluation completed",
            extra={
                "extra_data": {
                    "groundedness_score": result.groundedness_score,
                    "unsupported_statement_count": len(result.unsupported_statements),
                }
            },
        )

        return result

    except Exception as e:
        logger.warning(
            "Groundedness evaluation parsing failed",
            extra={"extra_data": {"error": str(e), "raw_response": raw_response[:1000]}},
        )

        return GroundednessResult(
            groundedness_score=None,
            explanation=f"Failed to parse JSON response: {raw_response}",
            unsupported_statements=[],
        )