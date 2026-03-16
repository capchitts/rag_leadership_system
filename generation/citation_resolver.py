import re


def build_chunk_lookup(chunks):
    lookup = {}

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        metadata = chunk.get("metadata", {})

        lookup[chunk_id] = {
            "source": metadata.get("source", "unknown"),
            "page": metadata.get("page", "unknown"),
            "section": metadata.get("section", "general"),
        }

    return lookup


def extract_cited_chunk_ids(text):
    pattern = r"\[([^\[\]]+?)\]"
    matches = re.findall(pattern, text)

    chunk_ids = []
    for match in matches:
        parts = [p.strip() for p in match.split("|")]
        if parts:
            chunk_ids.append(parts[0])

    return chunk_ids


def append_citation_summary(answer_text, chunks):
    lookup = build_chunk_lookup(chunks)
    cited_ids = extract_cited_chunk_ids(answer_text)

    if not cited_ids:
        return answer_text

    lines = ["\n\nSources Used:"]
    seen = set()

    for cid in cited_ids:
        if cid in seen:
            continue
        seen.add(cid)

        meta = lookup.get(cid)
        if not meta:
            continue

        lines.append(
            f"- {cid}: {meta['source']}, page {meta['page']}, section {meta['section']}"
        )

    return answer_text + "\n" + "\n".join(lines)