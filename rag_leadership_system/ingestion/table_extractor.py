import re
import pdfplumber


def normalize_cell_text(value):
    if value is None:
        return ""

    text = str(value)
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\|{2,}", "|", text)
    return text.strip()


def normalize_header_text(value, idx):
    text = normalize_cell_text(value)
    return text if text else f"column_{idx}"


def is_meaningful_row(row):
    if not row:
        return False

    non_empty_cells = [cell for cell in row if cell]
    if len(non_empty_cells) < 2:
        return False

    joined = " ".join(non_empty_cells).strip()
    if len(joined) < 8:
        return False

    return True


def looks_like_malformed_table(headers, rows):
    """
    Heuristic filter for badly extracted tables.
    """
    if not rows:
        return True

    # Too many generic fallback headers usually means poor extraction
    fallback_headers = sum(1 for h in headers if h.startswith("column_"))
    if fallback_headers == len(headers):
        return True

    # If most rows have only one non-empty cell, table is probably broken
    sparse_rows = 0
    for row in rows:
        non_empty = sum(1 for c in row if c)
        if non_empty <= 1:
            sparse_rows += 1

    if rows and sparse_rows / len(rows) > 0.6:
        return True

    return False


def is_valid_table(headers, rows):
    if not headers or len(headers) < 2:
        return False

    non_empty_headers = [h for h in headers if h]
    if len(non_empty_headers) < 2:
        return False

    meaningful_rows = [row for row in rows if is_meaningful_row(row)]
    if not meaningful_rows:
        return False

    if looks_like_malformed_table(headers, meaningful_rows):
        return False

    return True


def extract_tables(pdf_path):
    extracted_tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()

            for table_idx, table in enumerate(tables):
                if not table or len(table) < 2:
                    continue

                raw_headers = table[0]
                raw_rows = table[1:]

                headers = [
                    normalize_header_text(h, i)
                    for i, h in enumerate(raw_headers)
                ]

                rows = []
                for raw_row in raw_rows:
                    if not raw_row:
                        continue

                    normalized_row = [normalize_cell_text(cell) for cell in raw_row]

                    # pad row if shorter than headers
                    if len(normalized_row) < len(headers):
                        normalized_row.extend([""] * (len(headers) - len(normalized_row)))

                    # trim row if longer than headers
                    normalized_row = normalized_row[:len(headers)]

                    if not is_meaningful_row(normalized_row):
                        continue

                    # skip repeated header row inside table body
                    if [c.lower() for c in normalized_row] == [h.lower() for h in headers]:
                        continue

                    rows.append(normalized_row)

                if not is_valid_table(headers, rows):
                    continue

                row_texts = []
                for row_idx, row in enumerate(rows, start=1):
                    pairs = []
                    for h, cell_value in zip(headers, row):
                        if cell_value:
                            pairs.append(f"{h}: {cell_value}")

                    if pairs:
                        row_text = f"Table row {row_idx} -> " + " | ".join(pairs)
                        row_texts.append(row_text)

                if not row_texts:
                    continue

                table_text = "\n".join(row_texts)

                extracted_tables.append({
                    "text": table_text,
                    "rows": row_texts,
                    "metadata": {
                        "page": page_num,
                        "type": "table",
                        "table_index": table_idx,
                        "row_count": len(rows),
                        "column_count": len(headers),
                        "headers": headers,
                        "extraction_method": "pdfplumber"
                    }
                })

    return extracted_tables