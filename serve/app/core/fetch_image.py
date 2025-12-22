from io import BytesIO

import httpx
from PIL import Image
from fastapi import HTTPException, status

MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


async def fetch_image_from_url(image_url: str) -> Image.Image:
    timeout = httpx.Timeout(5.0)

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        resp = await client.get(image_url)

    if resp.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to fetch image from image_url",
        )

    content_type = resp.headers.get("content-type", "").split(";")[0].lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image content-type: {content_type}",
        )

    content_length = len(resp.content)
    if content_length > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image size exceeds 5MB limit",
        )

    try:
        image = Image.open(BytesIO(resp.content)).convert("RGB")
        return image
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Invalid image data",
        )
