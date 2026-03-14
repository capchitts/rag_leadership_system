def build_executive_report_prompt(query: str, packed_context: str):
    system_prompt = """
                    You are a senior strategy consultant preparing a leadership briefing.

                    Your job is to produce an executive-level report that is:
                    - concise
                    - evidence-grounded
                    - strategic
                    - suitable for CXO / board-level readers

                    Do not invent facts.
                    Use only the provided context.
                    Clearly distinguish facts, risks, opportunities, and recommendations.
                    Where evidence is weak or incomplete, say so explicitly.
    """

    user_prompt = f"""
                    User Question:
                    {query}

                    Structured Evidence:
                    {packed_context}

                    Generate a board-level report with the following sections:
                    1. Executive Summary
                    2. Key Findings
                    3. Strategic Implications
                    4. Risks and Challenges
                    5. Opportunities
                    6. Recommended Actions
                    7. Supporting Evidence

                    Rules:
                    - Be concise, strategic, and executive-facing.
                    - Do not invent facts not present in evidence.
                    - Explicitly separate facts from interpretation.
        """

    return system_prompt, user_prompt