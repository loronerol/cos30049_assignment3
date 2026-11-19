"""Register, login and the current_user dependency."""
import hashlib
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import SECRET
from database import db
from schemas import Creds

router = APIRouter(prefix="/api")
bearer = HTTPBearer()


def hash_pw(pw: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 200_000).hex()


def make_token(uid: int, username: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(hours=12)
    return jwt.encode({"sub": str(uid), "name": username, "exp": exp}, SECRET, "HS256")


def current_user(cred: HTTPAuthorizationCredentials = Depends(bearer)) -> dict:
    try:
        p = jwt.decode(cred.credentials, SECRET, algorithms=["HS256"])
        return {"id": int(p["sub"]), "username": p["name"]}
    except jwt.PyJWTError:
        raise HTTPException(401, "Session expired. Please sign in again.")


@router.post("/register")
def register(c: Creds):
    salt = secrets.token_bytes(16)
    try:
        with db() as con:
            cur = con.execute(
                "insert into users(username,pw,salt) values(?,?,?)",
                (c.username, hash_pw(c.password, salt), salt.hex()),
            )
    except sqlite3.IntegrityError:
        raise HTTPException(409, "That username is taken.")
    return {"token": make_token(cur.lastrowid, c.username), "username": c.username}


@router.post("/login")
def login(c: Creds):
    with db() as con:
        u = con.execute("select * from users where username=?", (c.username,)).fetchone()
    if not u or not secrets.compare_digest(
        u["pw"], hash_pw(c.password, bytes.fromhex(u["salt"]))
    ):
        raise HTTPException(401, "Wrong username or password.")
    return {"token": make_token(u["id"], u["username"]), "username": u["username"]}
