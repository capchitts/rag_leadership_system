def build_context(chunks):

    context_parts = []

    for chunk in chunks:

        section = chunk["metadata"].get("section", "unknown")
        source = chunk["metadata"].get("source", "unknown")

        formatted = f"""
                    Source: {source}
                    Section: {section}

                    {chunk["text"]}
        """

        context_parts.append(formatted)

    return "\n\n".join(context_parts)