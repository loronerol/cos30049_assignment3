"""Settings, read from environment variables."""
import os
from pathlib import Path

SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
DB_PATH = os.getenv("DB_PATH", "app.db")
STATIC_DIR = Path(__file__).parent / "static"
