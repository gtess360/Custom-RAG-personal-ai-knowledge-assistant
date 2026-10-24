"""Configuration and path management for the Personal AI Knowledge Assistant.

This module provides validated absolute paths and loads environment variables
from a local .env file. Paths are computed relative to the project root so that
the application behaves identically regardless of the current working directory.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Derive the project root folder from this file's location:
# src/config.py -> parent is src/ -> parent of src/ is the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables from the root .env file if it exists.
# override=False preserves any explicitly passed system environment variables.
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=False)


@dataclass(frozen=True)
class ProjectPaths:
    """Immutable collection of standard absolute paths used throughout the project."""
    root: Path
    data_dir: Path
    storage_dir: Path
    evaluation_dir: Path
    docs_dir: Path


_storage_env = os.getenv("CHROMA_PERSIST_DIRECTORY", "").strip()
_data_env = os.getenv("DOCUMENTS_DIRECTORY", "").strip()

# Instantiate standard project paths (supporting persistent volume mounts in containers)
PATHS = ProjectPaths(
    root=PROJECT_ROOT,
    data_dir=Path(_data_env) if _data_env else PROJECT_ROOT / "data" / "documents",
    storage_dir=Path(_storage_env) if _storage_env else PROJECT_ROOT / "storage" / "chroma",
    evaluation_dir=PROJECT_ROOT / "evaluation",
    docs_dir=PROJECT_ROOT / "docs",
)

# Configurable OpenAI model identifier (defaults to gpt-4o-mini if unspecified)
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip() or DEFAULT_OPENAI_MODEL

# Default retrieval candidate volume (shared across backend, API, and evaluator defaults)
# Note: Frontend defaults are separately and statically configured; not dynamically shared.
DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5").strip() or "5")

# Allowed explicit application environments
VALID_APP_ENVS = frozenset({"development", "production"})


def validate_app_env(env_val: Optional[str] = None) -> str:
    """Validate that APP_ENV is explicitly set to 'development' or 'production'.

    Raises:
        ValueError: If APP_ENV is missing, empty, or set to an unrecognized value.
    """
    raw = env_val if env_val is not None else os.getenv("APP_ENV")
    if raw is None or not raw.strip():
        raise ValueError(
            "Missing required environment variable: APP_ENV.\n"
            "Action required: Explicitly set APP_ENV to 'development' or 'production' in your environment or '.env' file.\n"
            "For local development, add 'APP_ENV=development' to your root .env file."
        )
    cleaned = raw.strip().lower()
    if cleaned not in VALID_APP_ENVS:
        raise ValueError(
            f"Invalid APP_ENV value: '{raw}'.\n"
            f"Allowed values are strictly: {', '.join(sorted(VALID_APP_ENVS))}.\n"
            "Action required: Set APP_ENV=development for local pairing or APP_ENV=production for deployment."
        )
    return cleaned


# Application environment setting
APP_ENV: str = os.getenv("APP_ENV", "").strip().lower()


def ensure_directories() -> None:
    """Create essential project directories if they do not already exist."""
    PATHS.data_dir.mkdir(parents=True, exist_ok=True)
    PATHS.evaluation_dir.mkdir(parents=True, exist_ok=True)


def has_openai_api_key() -> bool:
    """Return True if an OpenAI API key is configured (non-empty and not the placeholder)."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    return bool(key) and key != "your_openai_api_key_here"


def get_openai_api_key() -> str:
    """Retrieve the OpenAI API key or raise an actionable error.
    
    This function should be called only when the OpenAI API is actually invoked
    (Phase 4+). During local setup and corpus creation, an API key is not required.
    
    Raises:
        ValueError: If OPENAI_API_KEY is missing or still set to the placeholder.
    """
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key or key == "your_openai_api_key_here":
        raise ValueError(
            "Missing or placeholder OPENAI_API_KEY.\n"
            "Action required: Open the local '.env' file in your project root "
            "and set OPENAI_API_KEY=<your_actual_key>.\n"
            "Note: Never commit .env to version control."
        )
    return key


def get_safe_config_summary() -> dict:
    """Return a dictionary of configuration settings safe for logging and display.
    
    Never exposes, prints, or logs the actual API key string.
    """
    return {
        "project_root": str(PATHS.root),
        "data_dir": str(PATHS.data_dir),
        "storage_dir": str(PATHS.storage_dir),
        "evaluation_dir": str(PATHS.evaluation_dir),
        "openai_model": OPENAI_MODEL,
        "openai_api_key_configured": has_openai_api_key(),
        "app_env": APP_ENV,
    }
