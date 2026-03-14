from sentence_transformers import SentenceTransformer
import numpy as np


class QueryExpander:

    def __init__(self):

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def expand(self, query):

        templates = [

            f"{query}",

            f"Key insights about {query}",

            f"Important information regarding {query}",

            f"Detailed explanation of {query}",

            f"Major issues related to {query}"
        ]

        return templates