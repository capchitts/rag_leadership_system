def build_query_expansion_prompt(query: str) -> str:
    return f"""
            You are a retrieval optimization assistant.

            Given the user query below, generate 5 alternative search queries
            that preserve meaning but vary wording, terminology, and phrasing.

            User Query:
            {query}

            Return only the rewritten queries as a numbered list.
    """