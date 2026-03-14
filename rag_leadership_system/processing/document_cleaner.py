import re
from typing import List, Dict, Any


def normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_page_artifacts(text: str) -> str:
    if not text:
        return ""

    lines = text.splitlines()
    cleaned_lines = []

    artifact_patterns = [
        r"^page\s+\d+(\s+of\s+\d+)?$",
        r"^\d+$",
        r"^confidential$",
        r"^internal use only$",
        r"^copyright.*$",
    ]

    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue

        is_artifact = any(
            re.match(pattern, stripped, flags=re.IGNORECASE)
            for pattern in artifact_patterns
        )

        if not is_artifact:
            cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines).strip()


def fix_broken_words(text: str) -> str:
    if not text:
        return ""

    # Example: "oper-\national" -> "operational"
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # Example: newline in middle of sentence -> space
    text = re.sub(r"(?<![\.\!\?\:])\n(?!\n)", " ", text)

    return text


def remove_repeated_headers_footers(
    pages: List[Dict[str, Any]],
    min_repetition_ratio: float = 0.6
) -> List[Dict[str, Any]]:
    """
    Detects repeated first/last lines across pages and removes them.
    Expected input:
    [
        {"text": "...", "metadata": {...}},
        ...
    ]
    """
    if not pages:
        return pages

    first_line_counts = {}
    last_line_counts = {}

    page_lines = []
    for page in pages:
        lines = [line.strip() for line in page["text"].splitlines() if line.strip()]
        page_lines.append(lines)

        if lines:
            first_line_counts[lines[0]] = first_line_counts.get(lines[0], 0) + 1
            last_line_counts[lines[-1]] = last_line_counts.get(lines[-1], 0) + 1

    threshold = max(2, int(len(pages) * min_repetition_ratio))

    repeated_first = {
        line for line, count in first_line_counts.items() if count >= threshold
    }
    repeated_last = {
        line for line, count in last_line_counts.items() if count >= threshold
    }

    cleaned_pages = []

    for page, lines in zip(pages, page_lines):
        new_lines = lines[:]

        if new_lines and new_lines[0] in repeated_first:
            new_lines = new_lines[1:]

        if new_lines and new_lines[-1] in repeated_last:
            new_lines = new_lines[:-1]

        cleaned_pages.append({
            "text": "\n".join(new_lines).strip(),
            "metadata": page["metadata"]
        })

    return cleaned_pages


def clean_text(text: str) -> str:
    text = normalize_whitespace(text)
    text = remove_page_artifacts(text)
    text = fix_broken_words(text)
    text = normalize_whitespace(text)
    return text


def clean_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Cleans a list of extracted page/table documents.
    """
    if not documents:
        return []

    text_docs = [doc for doc in documents if doc["metadata"].get("type") == "text"]
    non_text_docs = [doc for doc in documents if doc["metadata"].get("type") != "text"]

    text_docs = remove_repeated_headers_footers(text_docs)

    cleaned_text_docs = []
    for doc in text_docs:
        cleaned_text_docs.append({
            "text": clean_text(doc["text"]),
            "metadata": doc["metadata"]
        })

    cleaned_non_text_docs = []
    for doc in non_text_docs:
        cleaned_non_text_docs.append({
            "text": clean_text(doc["text"]),
            "metadata": doc["metadata"]
        })

    return cleaned_text_docs + cleaned_non_text_docs