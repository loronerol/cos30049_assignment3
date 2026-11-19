"""SQLite connection and table setup."""
import sqlite3

from config import DB_PATH


def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with db() as con:
        con.executescript(
            """
            create table if not exists users(
                id integer primary key, username text unique, pw text, salt text);
            create table if not exists reports(
                id integer primary key, user_id integer, sender text, subject text,
                body text, is_spam integer, confidence real, reason text,
                engine text, created text);
            """
        )
