"""Request bodies shared by the routes."""
from pydantic import BaseModel, Field


class Creds(BaseModel):
    username: str = Field(min_length=3, max_length=40)
    password: str = Field(min_length=6, max_length=128)


class Email(BaseModel):
    sender: str = Field(default="", max_length=200)
    subject: str = Field(default="", max_length=300)
    body: str = Field(min_length=1, max_length=20000)


class Bulk(BaseModel):
    emails: list[Email] = Field(min_length=1, max_length=25)


class GmailImport(BaseModel):
    address: str = Field(max_length=200)
    app_password: str = Field(max_length=100)
    folder: str = Field(default="INBOX", pattern=r"^(INBOX|\[Gmail\]/Spam)$")
    limit: int = Field(default=15, ge=1, le=25)
