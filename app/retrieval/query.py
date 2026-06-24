import re


CERTIFICATE_CODE_RE = re.compile(r"\bCERT-\d{4}-\d{3}\b", re.IGNORECASE)


def extract_certificate_code(question: str) -> str | None:
    match = CERTIFICATE_CODE_RE.search(question)
    return match.group(0).upper() if match else None
