import json
import re

from core.schemas import QueryExpansionResult
from prompts.query_expansion_prompt import build_query_expansion_prompt


def extract_json_text(raw_response: str) -> str:
    text = raw_response.strip()
    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No valid JSON found")

    return text[start:end + 1]


class LLMQueryExpander:
    def __init__(self, llm):
        self.llm = llm

    def expand(self, query: str):
        prompt = build_query_expansion_prompt(query)
        raw_response = self.llm.invoke(prompt)

        json_text = extract_json_text(raw_response)
        parsed = json.loads(json_text)

        result = QueryExpansionResult.model_validate(parsed)
        return result.expanded_queries