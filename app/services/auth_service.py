"""Password storage and JWT boundary; never log credentials or tokens."""
import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass

import jwt
from sqlalchemy import select

from app.config import get_settings
from app.db.mysql import get_mysql_database
from app.models import User

ROLES = {"sales", "coach", "admin"}
ITERATIONS = 600_000
TOKEN_SECONDS = 24 * 60 * 60


def hash_password(password):
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), ITERATIONS).hex()
    return f"pbkdf2_sha256${ITERATIONS}${digest}", salt


def verify_password(password, encoded, salt):
    try:
        algorithm, iterations, digest = encoded.split("$")
        if algorithm != "pbkdf2_sha256" or not 100_000 <= int(iterations) <= 2_000_000:
            return False
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(iterations)).hex()
        return hmac.compare_digest(actual, digest)
    except (ValueError, TypeError, AttributeError):
        return False


def jwt_secret():
    value = get_settings().auth.jwt_secret
    if value is None or len(value.get_secret_value().encode()) < 32:
        raise RuntimeError("APP_AUTH__JWT_SECRET must contain at least 32 bytes")
    return value.get_secret_value()


def issue_token(user):
    now = int(time.time())
    return jwt.encode({"sub": str(user.id), "role": user.role, "iat": now,
                       "exp": now + TOKEN_SECONDS}, jwt_secret(), algorithm="HS256")


def decode_token(token):
    claims = jwt.decode(token, jwt_secret(), algorithms=["HS256"],
                        options={"require": ["sub", "role", "iat", "exp"]})
    if (not isinstance(claims["sub"], str) or not claims["sub"].isascii()
            or not claims["sub"].isdigit() or int(claims["sub"]) <= 0
            or claims["role"] not in ROLES
            or type(claims["iat"]) is not int or type(claims["exp"]) is not int
            or claims["exp"] <= claims["iat"]
            or claims["exp"] - claims["iat"] > TOKEN_SECONDS
            or ("user_id" in claims and str(claims["user_id"]) != claims["sub"])):
        raise jwt.InvalidTokenError("invalid_claims")
    return claims


@dataclass(frozen=True)
class Principal:
    id: int
    username: str
    role: str
    enabled: bool = True


class AuthService:
    def __init__(self, session_factory=None):
        self.session_factory = session_factory or get_mysql_database().session_scope

    def get_user(self, user_id):
        with self.session_factory() as db:
            row = db.get(User, user_id)
            return Principal(row.id, row.username, row.role, row.enabled) if row else None

    def login(self, username, password):
        with self.session_factory() as db:
            row = db.execute(select(User).where(User.username == username)).scalar_one_or_none()
            # Unknown users take the same expensive password verification path.
            encoded = row.password_hash if row else f"pbkdf2_sha256${ITERATIONS}$" + "0" * 64
            salt = row.salt if row else "0" * 32
            valid = verify_password(password, encoded, salt)
            if not valid or not row or not row.enabled or row.role not in ROLES:
                return None
            return Principal(row.id, row.username, row.role, row.enabled)

    def can_view_sales(self, principal, target_id):
        if principal.id == target_id or principal.role == "admin":
            return True
        if principal.role != "coach":
            return False
        target = self.get_user(target_id)
        return bool(target and target.enabled and target.role == "sales" and
                    target.username in get_settings().auth.coach_sales.get(principal.username, []))
