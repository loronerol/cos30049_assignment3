"""The user's report: stats, 14-day trend and history."""
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends

from auth import current_user
from database import db

router = APIRouter(prefix="/api")


@router.get("/reports")
def reports(user: dict = Depends(current_user)):
    with db() as con:
        rows = [dict(r) for r in con.execute(
            "select id,sender,subject,is_spam,confidence,reason,engine,created "
            "from reports where user_id=? order by id desc", (user["id"],))]
    spam = sum(r["is_spam"] for r in rows)
    days = {}
    for r in rows:
        days.setdefault(r["created"][:10], [0, 0])[r["is_spam"]] += 1
    today = datetime.now(timezone.utc).date()
    trend = []
    for i in range(13, -1, -1):
        day = (today - timedelta(days=i)).isoformat()
        safe, bad = days.get(day, [0, 0])
        trend.append({"day": day, "safe": safe, "spam": bad})
    return {"username": user["username"], "total": len(rows), "spam": spam,
            "safe": len(rows) - spam, "items": rows, "trend": trend}


@router.delete("/reports/{rid}")
def delete_report(rid: int, user: dict = Depends(current_user)):
    with db() as con:
        con.execute("delete from reports where id=? and user_id=?", (rid, user["id"]))
    return {"ok": True}
