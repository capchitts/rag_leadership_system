def build_faithfulness_prompt(answer: str, context: str) -> str:
    return f"""
            You are evaluating whether an answer is fully supported by the provided context.

            Context:
            {context}

            Answer:
            {answer}

            Determine:
            1. Which claims are directly supported
            2. Which claims are partially supported
            3. Which claims are unsupported

            Return a structured assessment.
"""