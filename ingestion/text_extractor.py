import fitz
import re


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ ]+\n", "\n", text)
    return text.strip()


def is_meaningful_page_text(text: str, min_chars: int = 20) -> bool:
    if not text:
        return False

    cleaned = text.strip()
    if len(cleaned) < min_chars:
        return False

    alnum_count = len(re.findall(r"[A-Za-z0-9]", cleaned))
    return alnum_count >= 8


def extract_text(pdf_path):
    pages = []

    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            text = normalize_text(text)

            if not is_meaningful_page_text(text):
                continue

            pages.append((page_num, text))

    return pages