def build_groundedness_prompt(answer: str, context: str) -> str:
    return f"""
            You are evaluating groundedness of a generated answer.

            Context:
            {context}

            Answer:
            {answer}

            Score the answer on:
            - factual grounding
            - evidence alignment
            - hallucination risk

            Return:
            1. groundedness score out of 10
            2. explanation
            3. unsupported statements if any
        """