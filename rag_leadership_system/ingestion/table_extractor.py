import pdfplumber
import pandas as pd

def extract_tables(pdf_path):

    tables_text = []

    with pdfplumber.open(pdf_path) as pdf:

        for page in pdf.pages:

            tables = page.extract_tables()

            for table in tables:

                df = pd.DataFrame(table)

                tables_text.append(df.to_string())

    return tables_text