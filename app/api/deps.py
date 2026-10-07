"""HTTP and SSE share identity, role and resource checks before work starts."""
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.services.auth_service import AuthService, Principal, decode_token
from app.services.session_service import SessionService

bearer = HTTPBearer(auto_error=False)


def unauthorized():
    return HTTPException(401, "invalid_credentials", headers={"WWW-Authenticate": "Bearer"})


def get_auth_service():
    return AuthService()


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer),
                     service: AuthService = Depends(get_auth_service)) -> Principal:
    if credentials is None:
        raise unauthorized()
    try:
        claims = decode_token(credentials.credentials)
    except (jwt.InvalidTokenError, ValueError, TypeError):
        raise unauthorized() from None
    user = service.get_user(int(claims["sub"]))
    if not user or not user.enabled or user.role != claims["role"]:
        raise unauthorized()
    return user


def require_admin(user: Principal = Depends(get_current_user)):
    if user.role != "admin":
        raise HTTPException(403, "role_forbidden")
    return user


def require_reviewer(user: Principal = Depends(get_current_user)):
    if user.role not in {"coach", "admin"}:
        raise HTTPException(403, "role_forbidden")
    return user


def require_score_scope(user_id: int, user: Principal = Depends(get_current_user),
                        service: AuthService = Depends(get_auth_service)):
    if not service.can_view_sales(user, user_id):
        raise HTTPException(403, "sales_scope_forbidden")
    return user


def require_review_scope(user_id: int, user: Principal = Depends(require_reviewer),
                         service: AuthService = Depends(get_auth_service)):
    """M4 must supply the pending record's owner, never a caller's claimed owner."""
    if not service.can_view_sales(user, user_id):
        raise HTTPException(403, "sales_scope_forbidden")
    return user


def get_ownership_service():
    return SessionService()


def require_owned_session(session_id: str, user: Principal = Depends(get_current_user),
                          service: SessionService = Depends(get_ownership_service)):
    # Applies to mutations and lazy evaluation as well as streams/checkpoint reads.
    try:
        row = service.get_session(session_id, user_id=user.id)
    except PermissionError:
        raise HTTPException(403, "session_owner_forbidden") from None
    if row is None:
        raise HTTPException(404, "session_not_found")
    return user
