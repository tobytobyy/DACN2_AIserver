FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface

WORKDIR /app

# deps
COPY serve/requirements.txt /app/serve/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r /app/serve/requirements.txt

# source + artifacts
COPY serve /app/serve
COPY artifacts /app/artifacts

WORKDIR /app/serve
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]