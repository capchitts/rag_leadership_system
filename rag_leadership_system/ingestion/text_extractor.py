import fitz

def extract_text(pdf_path):

    doc = fitz.open(pdf_path)

    pages = []

    for page_num, page in enumerate(doc):

        text = page.get_text()

        pages.append((page_num + 1, text))

    return pages