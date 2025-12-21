from fastapi import HTTPException, UploadFile

MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5MB
ALLOWED_CT = {"image/jpeg", "image/png", "image/webp"}


async def read_image_bytes(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_CT:
        raise HTTPException(415, f"Unsupported content-type: {file.content_type}")

    data = await file.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image too large")
    return data
