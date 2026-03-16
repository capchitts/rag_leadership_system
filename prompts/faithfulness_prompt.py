def build_faithfulness_prompt(answer: str, context: str) -> str:
    return f"""
            You are a strict RAG evaluator.

            Task:
            Evaluate whether the answer is faithful to the provided context.

            Definitions:
            - Directly supported = explicitly stated in the context.
            - Partially supported = some evidence exists, but wording is stronger or broader than the evidence.
            - Unsupported = no evidence in the context.

            Context:
            {context}

            Answer:
            {answer}

            Instructions:
            - Break the answer into key claims.
            - Assess at most 8 key claims.
            - Use only the provided context.
            - Do not use outside knowledge.
            - Be strict and concise.
            

            Return ONLY valid JSON in this exact schema:
            {{
                "faithfulness_score": <integer 1-10>,
                "directly_supported_claims": [
                    "<claim 1>",
                    "<claim 2>"
                ],
                "partially_supported_claims": [
                    "<claim 1>",
                    "<claim 2>"
                ],
                "unsupported_claims": [
                    "<claim 1>",
                    "<claim 2>"
                ],
                "summary": "<short explanation>"
            }}

            Scoring:
            1 = mostly unsupported
            3 = weak faithfulness
            5 = mixed support
            7 = mostly faithful with minor overreach
            10 = fully faithful

            Do not wrap JSON in markdown fences.
            Do not include any text before or after the JSON.
            
        """.strip()