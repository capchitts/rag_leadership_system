def evaluate_faithfulness(llm, answer, context):

    prompt = f"""
                You are evaluating whether the answer is supported by the provided context.

                Context:
                {context}

                Answer:
                {answer}

                Question:
                Is the answer fully supported by the context?

                Respond with:
                Faithful
                or
                Not Faithful
            """

    response = llm.invoke(prompt)

    return response