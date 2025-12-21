from __future__ import annotations

from fastapi import HTTPException, UploadFile

# Giới hạn ảnh (tuỳ bạn chỉnh)
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5MB

# Chỉ cho phép các định dạng ảnh phổ biến
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


def validate_content_type(file: UploadFile) -> None:
    ct = (file.content_type or "").lower().strip()
    if ct not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content-type: {file.content_type}",
        )


async def read_upload_limited(
    file: UploadFile, *, max_bytes: int = MAX_IMAGE_BYTES
) -> bytes:
    """
    Đọc upload với hard limit.
    - Đọc tối đa max_bytes + 1
    - Nếu vượt quá => 413
    """
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large. Max {max_bytes} bytes.",
        )
    return data
