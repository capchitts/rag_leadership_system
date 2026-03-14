import re
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class QueryAnalysis:
    original_query: str
    normalized_query: str
    query_type: str
    should_expand: bool
    use_bm25: bool
    use_vector: bool
    bm25_weight: float
    vector_weight: float
    max_expansions: int
    reasons: List[str]


class QueryAnalyzer:
    """
    Analyze a query and decide the retrieval strategy.

    Query types:
    - exact_lookup
    - table_lookup
    - semantic_broad
    - natural_language_detailed
    - standard
    """

    EXACT_KEYWORDS = {
        "invoice",
        "invoice number",
        "policy number",
        "error code",
        "clause",
        "section",
        "page",
        "sku",
        "id",
        "reference number",
        "transaction id",
    }

    TABLE_KEYWORDS = {
        "table",
        "premium",
        "price",
        "pricing",
        "fee",
        "fees",
        "amount",
        "plan",
        "plans",
        "rate",
        "rates",
        "comparison",
    }

    SEMANTIC_HINTS = {
        "explain",
        "meaning",
        "overview",
        "benefits",
        "issues",
        "risks",
        "challenges",
        "insights",
        "summary",
        "difference",
        "differences",
        "why",
        "how",
        "what happens",
    }

    def analyze(self, query: str) -> QueryAnalysis:
        normalized = self._normalize(query)
        reasons: List[str] = []

        word_count = len(normalized.split())
        has_year = bool(re.search(r"\b(19|20)\d{2}\b", normalized))
        has_number = bool(re.search(r"\b\d+(?:\.\d+)?\b", normalized))
        has_quoted_text = '"' in query or "'" in query
        has_code_like_token = bool(re.search(r"\b[a-zA-Z]+[-_][a-zA-Z0-9_-]+\b", query)) or bool(
            re.search(r"\b[a-zA-Z0-9]{6,}\b", query)
        )

        exact_keyword_match = any(k in normalized for k in self.EXACT_KEYWORDS)
        table_keyword_match = any(k in normalized for k in self.TABLE_KEYWORDS)
        semantic_hint_match = any(k in normalized for k in self.SEMANTIC_HINTS)

        # -------- classify query --------
        if table_keyword_match and (has_number or has_year or "plan" in normalized):
            query_type = "table_lookup"
            reasons.append("Detected pricing/table-style query.")
        elif exact_keyword_match or has_quoted_text or has_code_like_token:
            query_type = "exact_lookup"
            reasons.append("Detected exact/structured lookup query.")
        elif word_count <= 4 and not has_number:
            query_type = "semantic_broad"
            reasons.append("Very short query without numeric constraints.")
        elif word_count >= 10 or semantic_hint_match:
            query_type = "natural_language_detailed"
            reasons.append("Detailed natural-language question.")
        else:
            query_type = "standard"
            reasons.append("General balanced retrieval query.")

        # -------- decide retrieval strategy --------
        if query_type == "exact_lookup":
            should_expand = False
            use_bm25 = True
            use_vector = True
            bm25_weight = 0.70
            vector_weight = 0.30
            max_expansions = 1
            reasons.append("Favor BM25 due to exact-match intent.")

        elif query_type == "table_lookup":
            should_expand = False
            use_bm25 = True
            use_vector = True
            bm25_weight = 0.65
            vector_weight = 0.35
            max_expansions = 1
            reasons.append("Favor lexical retrieval for table/field lookup.")

        elif query_type == "semantic_broad":
            should_expand = True
            use_bm25 = True
            use_vector = True
            bm25_weight = 0.30
            vector_weight = 0.70
            max_expansions = 3
            reasons.append("Allow expansion for broad semantic recall.")

        elif query_type == "natural_language_detailed":
            should_expand = False
            use_bm25 = True
            use_vector = True
            bm25_weight = 0.45
            vector_weight = 0.55
            max_expansions = 1
            reasons.append("Detailed query already contains enough semantics.")

        else:  # standard
            should_expand = True if word_count <= 6 and not has_year and not has_number else False
            use_bm25 = True
            use_vector = True
            bm25_weight = 0.20
            vector_weight = 0.80
            max_expansions = 2 if should_expand else 1
            reasons.append("Balanced hybrid retrieval strategy.")

        return QueryAnalysis(
            original_query=query,
            normalized_query=normalized,
            query_type=query_type,
            should_expand=should_expand,
            use_bm25=use_bm25,
            use_vector=use_vector,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
            max_expansions=max_expansions,
            reasons=reasons,
        )

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def as_dict(self, analysis: QueryAnalysis) -> Dict:
        return {
            "original_query": analysis.original_query,
            "normalized_query": analysis.normalized_query,
            "query_type": analysis.query_type,
            "should_expand": analysis.should_expand,
            "use_bm25": analysis.use_bm25,
            "use_vector": analysis.use_vector,
            "bm25_weight": analysis.bm25_weight,
            "vector_weight": analysis.vector_weight,
            "max_expansions": analysis.max_expansions,
            "reasons": analysis.reasons,
        }