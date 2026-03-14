import re

class DocumentCleaner:
    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""

        # Remove broken hyphenation across lines: "differen-\ntiation" -> "differentiation"
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

        # Replace remaining newlines with spaces where appropriate
        text = re.sub(r"\n+", "\n", text)
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

        # Remove excessive whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove repeated stray numeric index artifacts
        text = re.sub(r"\b\d+\s+\d+\s+\d+\b", " ", text)

        # Strip page-only junk
        text = re.sub(r"^\s*Page\s+\d+\s*$", "", text, flags=re.IGNORECASE | re.MULTILINE)

        return text.strip()