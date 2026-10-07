"""Read-only startup prerequisites without PowerShell inline-code quoting."""


def main() -> int:
    try:
        import uvicorn  # noqa: F401 -- required for the application launcher
        from app.config import get_settings
        from app.services.auth_service import jwt_secret

        get_settings()
        jwt_secret()
    except Exception as exc:
        # Configuration exceptions can contain credentials. Print the type only.
        print(f"[FAIL] Python/config/JWT prerequisites: {type(exc).__name__}")
        return 1
    print("[OK] Python/config/JWT prerequisites")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
