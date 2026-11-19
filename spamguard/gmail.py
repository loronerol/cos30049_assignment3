"""Import emails from Gmail over IMAP (app password, nothing stored)."""
import email
import imaplib
import re
from email.header import decode_header, make_header

from fastapi import APIRouter, Depends, HTTPException

from analyze import check_many
from auth import current_user
from schemas import Email, GmailImport

router = APIRouter(prefix="/api")


def _text(msg) -> str:
    plain = html = ""
    for part in msg.walk():
        if part.get_content_maintype() != "text" or part.get_filename():
            continue
        raw = part.get_payload(decode=True) or b""
        txt = raw.decode(part.get_content_charset() or "utf-8", "replace")
        if part.get_content_type() == "text/html":
            html = html or re.sub(r"<[^>]+>", " ", txt)
        else:
            plain = plain or txt
    return (plain or html).strip()


@router.post("/gmail/import")
def gmail_import(g: GmailImport, user: dict = Depends(current_user)):
    try:
        box = imaplib.IMAP4_SSL("imap.gmail.com")
        box.login(g.address, g.app_password.replace(" ", ""))
    except (imaplib.IMAP4.error, OSError):
        raise HTTPException(400, "Gmail rejected the login. Use a Google app password and turn on IMAP.")
    emails = []
    try:
        box.select(f'"{g.folder}"', readonly=True)
        _, data = box.search(None, "ALL")
        for num in data[0].split()[-g.limit:][::-1]:
            _, parts = box.fetch(num, "(BODY.PEEK[])")
            msg = email.message_from_bytes(parts[0][1])
            subj = str(make_header(decode_header(msg.get("Subject", ""))))
            emails.append(Email(sender=msg.get("From", "")[:200], subject=subj[:300],
                                body=(_text(msg) or "(empty)")[:20000]))
    finally:
        try:
            box.logout()
        except Exception:
            pass
    if not emails:
        raise HTTPException(404, "No emails found in that folder.")
    return {"results": check_many(user["id"], emails)}
