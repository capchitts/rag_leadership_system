def build_groundedness_prompt(answer: str, context: str) -> str:
    return f"""
            You are a strict RAG evaluator.

            Task:
            Evaluate how well the answer is grounded in the provided context.

            Definitions:
            - Grounded = claims in the answer are supported by the context.
            - Ungrounded = claims are missing from the context, exaggerated, or inferred too strongly.

            Context:
            {context}

            Answer:
            {answer}

            Instructions:
            - Use only the provided context.
            - Do not use outside knowledge.
            - Be strict.
            - If a claim is only partially supported, treat it as partially grounded.
            

            Return ONLY valid JSON in this exact schema:
            {{
            "groundedness_score": <integer 1-10>,
            "explanation": "<short explanation>",
            "unsupported_statements": [
                "<statement 1>",
                "<statement 2>"
            ]
            }}

            Scoring:
            1 = mostly unsupported / hallucinated
            3 = weakly grounded
            5 = partially grounded
            7 = mostly grounded with minor unsupported inference
            10 = fully grounded

            Do not wrap JSON in markdown fences.
            Do not include any text before or after the JSON.
            
        """.strip()