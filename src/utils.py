import re

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # remove special chars
    return text.strip()


def get_reasons(text: str):
    reasons = []
    t = text.lower()

    if "no experience" in t:
        reasons.append("Unrealistic requirement: no experience needed")

    if "earn" in t and "$" in t:
        reasons.append("Suspicious salary mention")

    if "work from home" in t:
        reasons.append("Common scam phrase: work from home")

    if "quick money" in t:
        reasons.append("Too-good-to-be-true phrasing")

    if "no interview" in t:
        reasons.append("Suspicious hiring process: no interview")

    return reasons