def evaluate_groundedness(llm, answer, context):

    prompt = f"""
            Determine whether the answer is grounded in the provided evidence.

            Context:
            {context}

            Answer:
            {answer}

            Score groundedness from 1 to 5.

            1 = Not grounded
            5 = Fully grounded

            Return only the score.
        """

    response = llm.invoke(prompt)

    return response