from app.core.config import settings
from fastapi import Header, HTTPException, status


# if ENV=dev/local: bypass security
# if ENV!=dev/local: enforce security by checking INTERNAL_TOKEN in headers
# if INTERNAL_TOKEN is not set: 500 error (misconfiguration)


async def verify_internal_token(x_internal_token: str = Header(default="")) -> None:
    # Dev bypass
    if settings.is_dev:
        return

    # Non-dev must have server token configured
    if not settings.INTERNAL_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server misconfigured: INTERNAL_TOKEN is empty",
        )

    # Require matching header
    if not x_internal_token or x_internal_token != settings.INTERNAL_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )
