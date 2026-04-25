import base64
import hashlib
import hmac
import json
import os
import time
from datetime import datetime
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pymongo import ASCENDING

from app.clients.mongo_history_utils import get_history_mongo_tool


AUTH_SECRET_KEY = os.getenv("AUTH_SECRET_KEY", os.getenv("JWT_SECRET_KEY", "change-this-secret"))
AUTH_TOKEN_EXPIRE_SECONDS = int(os.getenv("AUTH_TOKEN_EXPIRE_SECONDS", str(24 * 60 * 60)))
ADMIN_INVITE_CODE = os.getenv("ADMIN_INVITE_CODE", "admin")
PBKDF2_ROUNDS = int(os.getenv("PASSWORD_PBKDF2_ROUNDS", "310000"))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def _urlsafe_b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _urlsafe_b64decode(raw: str) -> bytes:
    padding = "=" * (-len(raw) % 4)
    return base64.urlsafe_b64decode((raw + padding).encode("ascii"))


def _users_collection():
    mongo_tool = get_history_mongo_tool()
    users = mongo_tool.db["users"]
    users.create_index([("username", ASCENDING)], unique=True)
    return users


def get_password_hash(password: str) -> str:
    if not password:
        raise ValueError("password is required")
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS)
    return "pbkdf2_sha256${}${}${}".format(
        PBKDF2_ROUNDS,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(digest).decode("ascii"),
    )


def verify_password(password: str, password_hash: str) -> bool:
    if not password or not password_hash:
        return False
    try:
        scheme, rounds, salt_b64, digest_b64 = password_hash.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        salt = base64.b64decode(salt_b64.encode("ascii"))
        expected = base64.b64decode(digest_b64.encode("ascii"))
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(rounds))
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def create_access_token(username: str, role: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": username,
        "role": role,
        "exp": int(time.time()) + AUTH_TOKEN_EXPIRE_SECONDS,
    }
    signing_input = "{}.{}".format(
        _urlsafe_b64encode(json.dumps(header, separators=(",", ":")).encode("utf-8")),
        _urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")),
    )
    signature = hmac.new(AUTH_SECRET_KEY.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
    return f"{signing_input}.{_urlsafe_b64encode(signature)}"


def decode_access_token(token: str) -> dict:
    try:
        header_b64, payload_b64, signature_b64 = token.split(".", 2)
        signing_input = f"{header_b64}.{payload_b64}"
        expected = hmac.new(
            AUTH_SECRET_KEY.encode("utf-8"),
            signing_input.encode("ascii"),
            hashlib.sha256,
        ).digest()
        actual = _urlsafe_b64decode(signature_b64)
        if not hmac.compare_digest(actual, expected):
            raise ValueError("invalid signature")
        payload = json.loads(_urlsafe_b64decode(payload_b64).decode("utf-8"))
        if int(payload.get("exp", 0)) < int(time.time()):
            raise ValueError("token expired")
        return payload
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效或过期的认证令牌",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def resolve_role(role: Optional[str], admin_code: Optional[str]) -> str:
    requested = (role or "user").strip().lower()
    if requested != "admin":
        return "user"
    if admin_code and admin_code == ADMIN_INVITE_CODE:
        return "admin"
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="管理员邀请码错误")


def register_user(username: str, password: str, role: Optional[str] = None, admin_code: Optional[str] = None) -> dict:
    username = (username or "").strip()
    password = (password or "").strip()
    if not username or not password:
        raise HTTPException(status_code=400, detail="用户名和密码不能为空")

    users = _users_collection()
    if users.find_one({"username": username}):
        raise HTTPException(status_code=409, detail="用户名已存在")

    resolved_role = resolve_role(role, admin_code)
    user = {
        "username": username,
        "password_hash": get_password_hash(password),
        "role": resolved_role,
        "created_at": datetime.utcnow(),
    }
    users.insert_one(user)
    return {"username": username, "role": resolved_role}


def authenticate_user(username: str, password: str) -> Optional[dict]:
    users = _users_collection()
    user = users.find_one({"username": (username or "").strip()})
    if not user or not verify_password(password or "", user.get("password_hash", "")):
        return None
    return {"username": user["username"], "role": user.get("role", "user")}


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    payload = decode_access_token(token)
    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="无效或过期的认证令牌")
    users = _users_collection()
    user = users.find_one({"username": username})
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="无效或过期的认证令牌")
    return {"username": user["username"], "role": user.get("role", "user")}


def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="管理员权限不足")
    return current_user
