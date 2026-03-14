def build_embedding_text(chunk):

    metadata = chunk["metadata"]

    section = metadata.get("section", "general")
    source = metadata.get("source", "")
    page = metadata.get("page", "")

    text = chunk["text"]

    enriched_text = f"""
    Document: {source}
    Section: {section}
    Page: {page}

    Content:
    {text}
    """

    return enriched_text.strip()