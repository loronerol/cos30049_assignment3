"""Check one email or many, and save the results."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from auth import current_user
from classifier import classify
from database import db
from schemas import Bulk, Email

router = APIRouter(prefix="/api")


def check_many(uid: int, emails: list) -> list:
    """Classify emails (5 at a time) and store them in the user's report."""
    with ThreadPoolExecutor(max_workers=5) as ex:
        results = list(ex.map(lambda m: classify(m.sender, m.subject, m.body), emails))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with db() as con:
        con.executemany(
            "insert into reports(user_id,sender,subject,body,is_spam,confidence,reason,engine,created)"
            " values(?,?,?,?,?,?,?,?,?)",
            [(uid, m.sender, m.subject, m.body[:500], int(r["is_spam"]), r["confidence"],
              r["reason"], r["engine"], now) for m, r in zip(emails, results)])
    return [{"sender": m.sender, "subject": m.subject, **r} for m, r in zip(emails, results)]


@router.post("/analyze")
def analyze(e: Email, user: dict = Depends(current_user)):
    return check_many(user["id"], [e])[0]


@router.post("/analyze/bulk")
def analyze_bulk(b: Bulk, user: dict = Depends(current_user)):
    return {"results": check_many(user["id"], b.emails)}
