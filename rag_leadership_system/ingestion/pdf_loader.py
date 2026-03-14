import os
from ingestion.text_extractor import extract_text
from ingestion.table_extractor import extract_tables

def load_pdf_documents(folder):

    documents = []

    for file in os.listdir(folder):

        if file.endswith(".pdf"):

            path = os.path.join(folder, file)

            text_pages = extract_text(path)

            tables = extract_tables(path)

            for page_num, text in text_pages:

                documents.append({
                    "text": text,
                    "metadata": {
                        "source": file,
                        "page": page_num,
                        "type": "text"
                    }
                })

            for table in tables:

                documents.append({
                    "text": table,
                    "metadata": {
                        "source": file,
                        "type": "table"
                    }
                })

    return documents