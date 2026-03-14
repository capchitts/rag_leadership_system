from collections import defaultdict


def pack_context(chunks):

    section_map = defaultdict(list)

    for chunk in chunks:

        section = chunk["metadata"].get("section", "general")

        section_map[section].append(chunk["text"])


    structured_context = ""

    for section, texts in section_map.items():

        structured_context += f"\n=== {section.upper()} ===\n"

        for t in texts:

            structured_context += f"- {t}\n\n"


    return structured_context