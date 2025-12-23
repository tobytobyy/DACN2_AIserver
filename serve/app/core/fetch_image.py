from io import BytesIO

import httpx
from PIL import Image
from fastapi import HTTPException, status

MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

CHUNK_SIZE = 64 * 1024  # 64KB


async def fetch_image_from_url(image_url: str) -> Image.Image:
    """
    Stream download with a hard byte cap (<= 5MB).
    NOTE: SSRF hardening is intentionally NOT included (per current requirement).
    """
    timeout = httpx.Timeout(10.0)

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        async with client.stream("GET", image_url) as resp:
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to fetch image from image_url.",
                )

            content_type = resp.headers.get("content-type", "").split(";")[0].lower()
            if content_type not in ALLOWED_CONTENT_TYPES:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=f"Unsupported image content-type: {content_type}",
                )

            # Early reject if Content-Length provided and too large
            cl = resp.headers.get("content-length")
            if cl:
                try:
                    if int(cl) > MAX_IMAGE_SIZE:
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail="Image size exceeds 5MB limit.",
                        )
                except ValueError:
                    # ignore invalid content-length; enforce via streaming cap below
                    pass

            buf = BytesIO()
            total = 0

            async for chunk in resp.aiter_bytes(CHUNK_SIZE):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_IMAGE_SIZE:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="Image size exceeds 5MB limit.",
                    )
                buf.write(chunk)

    try:
        return Image.open(BytesIO(buf.getvalue())).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Invalid image data.",
        )
