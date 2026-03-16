def build_query_expansion_prompt(query: str) -> str:
    return f"""
You are a retrieval optimization assistant.

Given the user query below, generate 5 alternative search queries
that preserve meaning but vary wording, terminology, and phrasing.

User Query:
{query}

Return ONLY valid JSON in this exact schema:
{{
  "expanded_queries": [
    "<query 1>",
    "<query 2>",
    "<query 3>",
    "<query 4>",
    "<query 5>"
  ]
}}

Rules:
- Preserve the original meaning.
- Keep queries retrieval-friendly.
- Do not add explanation.
- Do not wrap JSON in markdown fences.
- Do not include any text before or after the JSON.
""".strip()