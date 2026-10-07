from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, SecretStr

from app.api.deps import get_auth_service, get_current_user, unauthorized
from app.services.auth_service import AuthService, Principal, TOKEN_SECONDS, issue_token

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: SecretStr = Field(min_length=1, max_length=1024)


@router.post("/login")
def login(payload: LoginRequest, service: AuthService = Depends(get_auth_service)):
    user = service.login(payload.username, payload.password.get_secret_value())
    if user is None:
        raise unauthorized()
    return {"access_token": issue_token(user), "token_type": "bearer",
            "expires_in": TOKEN_SECONDS, "user": {"id": user.id, "username": user.username, "role": user.role}}


@router.get("/me")
def me(user: Principal = Depends(get_current_user)):
    return {"id": user.id, "username": user.username, "role": user.role}
