import re


CERTIFICATE_CODE_RE = re.compile(
    r"\b(?:CERT-\d{4}-\d{3,}|[A-Z]{1,6}-\d{3,4}[.-]\d{2}[.-]\d{1,3})\b",
    re.IGNORECASE,
)


def extract_certificate_code(question: str) -> str | None:
    match = CERTIFICATE_CODE_RE.search(question)
    return match.group(0).upper() if match else None


def extract_customer_id(question: str) -> int | None:
    """Extract a customer ID from a question like "cliente 8" or "customer 202".

    Args:
        question: The user's question text.

    Returns:
        The integer customer ID if a known pattern is found, otherwise None.
    """
    match = re.search(r"\b(?:cliente|customer|client)\s+(\d+)\b", question, re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))
