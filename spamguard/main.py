"""Entry point: uvicorn main:app --reload"""
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import analyze, auth, gmail, reports
from config import STATIC_DIR
from database import init_db

init_db()
app = FastAPI(title="SpamGuard")
for module in (auth, analyze, gmail, reports):
    app.include_router(module.router)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
