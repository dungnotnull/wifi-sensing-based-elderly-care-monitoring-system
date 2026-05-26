"""
JWT-based authentication for ElderCare Dashboard.

Provides:
  - Password hashing (SHA-256 + salt)
  - JWT token creation and verification (python-jose)
  - FastAPI dependency for protected routes
  - Token-based login/logout flow

Default credentials are set via env vars or config.
"""

import hashlib
import hmac
import logging
import os
import secrets
from datetime import datetime, timedelta
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

load_dotenv()

logger = logging.getLogger(__name__)

SECRET_KEY = os.getenv("DASHBOARD_SECRET_KEY", "eldercare-secret-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = int(os.getenv("DASHBOARD_TOKEN_EXPIRE_HOURS", "24"))

DEFAULT_USERNAME = os.getenv("DASHBOARD_USERNAME", "admin")
DEFAULT_PASSWORD_HASH = os.getenv("DASHBOARD_PASSWORD_HASH")
_salt = os.getenv("DASHBOARD_SALT", "eldercare-dashboard-salt")

_security = HTTPBearer(auto_error=False)


def _hash_password(password: str) -> str:
    return hashlib.sha256((_salt + password).encode()).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return hmac.compare_digest(_hash_password(plain_password), hashed_password)


_DEFAULT_HASH = DEFAULT_PASSWORD_HASH or _hash_password(
    os.getenv("DASHBOARD_PASSWORD", "eldercare")
)


def create_access_token(username: str, expires_delta: Optional[timedelta] = None) -> str:
    expire = datetime.utcnow() + (expires_delta or timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS))
    to_encode = {"sub": username, "exp": expire, "iat": datetime.utcnow()}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: Optional[str] = payload.get("sub")
        return username
    except JWTError:
        return None


def authenticate_user(username: str, password: str) -> Optional[str]:
    if username != DEFAULT_USERNAME:
        return None
    if not verify_password(password, _DEFAULT_HASH):
        return None
    return create_access_token(username)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_security),
) -> str:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please log in.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    username = decode_token(credentials.credentials)
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return username


def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_security),
) -> Optional[str]:
    if credentials is None:
        return None
    return decode_token(credentials.credentials)
