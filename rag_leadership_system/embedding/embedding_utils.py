def build_embedding_text(chunk):

    metadata = chunk.metadata

    section = metadata.section or "general"
    source = metadata.source or ""
    page = metadata.page or ""

    text = chunk.text

    enriched_text = f"""
    Document: {source}
    Section: {section}
    Page: {page}

    Content:
    {text}
    """

    return enriched_text.strip()