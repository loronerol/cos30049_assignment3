"""The AI model. Edit this file to change the prompt, model or provider."""
import json
import os

from config import MODEL

SYSTEM = (
    "You are an email spam and phishing classifier. Reply with JSON only: "
    '{"is_spam": true|false, "confidence": 0-1, "reason": "one short sentence"}'
)
SPAM_WORDS = ["winner", "free", "urgent", "click here", "verify your account", "lottery",
              "password", "bitcoin", "limited time", "act now", "prize", "wire transfer",
              "unsubscribe", "100% guaranteed", "claim"]


def heuristic(sender: str, subject: str, body: str) -> dict:
    """Keyword fallback used when no API key is set or the AI call fails."""
    text = f"{subject} {body}".lower()
    hits = [w for w in SPAM_WORDS if w in text]
    score = min(1.0, 0.25 * len(hits) + (0.2 if "http" in text else 0))
    return {
        "is_spam": score >= 0.5,
        "confidence": round(max(score, 1 - score), 2),
        "reason": f"Matched suspicious phrases: {', '.join(hits)}." if hits
        else "No common spam phrases found.",
        "engine": "rules",
    }


def classify(sender: str, subject: str, body: str) -> dict:
    if not os.getenv("ANTHROPIC_API_KEY"):
        return heuristic(sender, subject, body)
    try:
        import anthropic

        msg = anthropic.Anthropic().messages.create(
            model=MODEL,
            max_tokens=300,
            system=SYSTEM,
            messages=[{"role": "user",
                       "content": f"From: {sender}\nSubject: {subject}\n\n{body[:6000]}"}],
        )
        text = msg.content[0].text.strip().strip("`").removeprefix("json").strip()
        r = json.loads(text)
        return {"is_spam": bool(r["is_spam"]), "confidence": float(r["confidence"]),
                "reason": str(r["reason"]), "engine": MODEL}
    except Exception as e:
        r = heuristic(sender, subject, body)
        r["reason"] += f" (AI unavailable: {type(e).__name__})"
        return r
