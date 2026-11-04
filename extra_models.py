import json
import os
import re
import requests

def factcheck_score(text: str) -> float:
    r = requests.get(
        "https://factchecktools.googleapis.com/v1alpha1/claims:search",
        params={"query": text[:300], "key": os.environ.get("FACTCHECK_API_KEY", ""), "pageSize": 5},
        timeout=10,
    )
    r.raise_for_status()
    ratings = []
    for claim in r.json().get("claims", []):
        for review in claim.get("claimReview", []):
            ratings.append(review.get("textualRating", "").lower())
    if not ratings:
        return 0.5
    bad = sum(bool(re.search(r"false|fake|misleading|incorrect|hoax|no evidence|pants", x)) for x in ratings)
    good = sum(bool(re.search(r"\btrue\b|correct|accurate", x)) and not re.search(r"false|not true|partly", x) for x in ratings)
    if bad > good:
        return 0.9
    if good > bad:
        return 0.1
    return 0.5

def llm_score(text: str) -> float:
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=os.environ.get("JUDGE_MODEL", "claude-3-haiku-20240307"),
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": (
                "You are a careful fact-checker. Estimate the probability that the claim below is false. "
                'Reply with JSON only, like {"p_false": 0.82}. If you cannot tell, use a value near 0.5.\n\n'
                f"Claim: {text}"
            ),
        }],
    )
    match = re.search(r"\{.*\}", msg.content[0].text, re.S)
    return float(json.loads(match.group(0))["p_false"])
