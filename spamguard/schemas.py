"""Request bodies shared by the routes."""
from pydantic import BaseModel, ConfigDict, Field


class Creds(BaseModel):
    username: str = Field(min_length=3, max_length=40)
    password: str = Field(min_length=6, max_length=128)


class Email(BaseModel):
    sender: str = Field(default="", max_length=200)
    subject: str = Field(default="", max_length=300)
    body: str = Field(min_length=1, max_length=20000)


class AccountFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retweets_received_per_tweet_mean: float = Field(ge=0, allow_inf_nan=False)
    frac_replies: float = Field(ge=0, le=1, allow_inf_nan=False)
    followers_count: int = Field(ge=0)
    followees_count: int = Field(ge=0)
    statuses_count: int = Field(ge=0)
    account_age_days: float = Field(ge=0, allow_inf_nan=False)
    default_profile: int = Field(ge=0, le=1)
    has_description: int = Field(ge=0, le=1)
    description_length: int = Field(ge=0)


class Bulk(BaseModel):
    emails: list[Email] = Field(min_length=1, max_length=25)


class GmailImport(BaseModel):
    address: str = Field(max_length=200)
    app_password: str = Field(max_length=100)
    folder: str = Field(default="INBOX", pattern=r"^(INBOX|\[Gmail\]/Spam)$")
    limit: int = Field(default=15, ge=1, le=25)
