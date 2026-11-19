"""Entry point: uvicorn main:app --reload"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import auth
from config import STATIC_DIR
from database import init_db
from social_accounts import load_bundle, router as social_accounts_router

init_db()


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_bundle()
    yield


app = FastAPI(title="SpamGuard", lifespan=lifespan)
app.include_router(auth.router)
app.include_router(social_accounts_router)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
