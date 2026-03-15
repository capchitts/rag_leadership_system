import re
import json

from prompts.faithfulness_prompt import build_faithfulness_prompt
from core.schemas import FaithfulnessResult
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


def evaluate_faithfulness(llm, answer: str, context: str) -> FaithfulnessResult:
    logger.info(
        "Faithfulness evaluation started",
        extra={"extra_data": {"answer_chars": len(answer), "context_chars": len(context)}},
    )

    prompt = build_faithfulness_prompt(answer=answer, context=context)
    raw_response = llm.invoke(prompt)

    try:
        json_text = _extract_json(raw_response)
        parsed = json.loads(json_text)

        result = FaithfulnessResult.model_validate(parsed)

        logger.info(
            "Faithfulness evaluation completed",
            extra={
                "extra_data": {
                    "faithfulness_score": result.faithfulness_score,
                    "direct_supported_count": len(result.directly_supported_claims),
                    "partial_supported_count": len(result.partially_supported_claims),
                    "unsupported_count": len(result.unsupported_claims),
                }
            },
        )

        return result

    except Exception as e:
        logger.warning(
            "Faithfulness evaluation parsing failed",
            extra={"extra_data": {"error": str(e), "raw_response": raw_response[:1000]}},
        )

        return FaithfulnessResult(
            faithfulness_score=None,
            directly_supported_claims=[],
            partially_supported_claims=[],
            unsupported_claims=[],
            summary=f"Failed to parse JSON response: {raw_response}",
        )