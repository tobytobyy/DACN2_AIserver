# DACN2 AI Server

This project is a specialized AI backend server designed to process multimodal inputs (text and images). It features
intelligent routing to specific domain pipelines (Food and Health) using Vision models (CLIP, BLIP) and Large Language
Models (Ollama).

## Key Features

- **Multimodal Inference**: Accepts both text messages and image URLs.
- **Intelligent Routing**: Automatically detects if an image is related to Food or Health domains using CLIP/Vision
  Router.
- **Domain-Specific Pipelines**:
    - **Food Pipeline**: Analyzes food images for nutritional info, recipes, etc.
    - **Health Pipeline**: Analyzes medical/health-related images (e.g., prescriptions, symptoms).
- **LLM Integration**: Powered by Ollama for natural language generation and reasoning.
- **Security**: Protected by `X-Internal-Token` for internal service communication.
- **Localization**: Automatic language detection (Vietnamese/English) and response localization.

## Project Structure

```text
serve/app/
├── main.py                 # Entry point
├── api/v1/routes/          # API Routes
│   └── inference.py        # Chat endpoint
├── core/                   # Core utilities (Config, Security, Logging)
├── domain/                 # Business Logic & Pipelines (Food, Health, LLM)
├── infra/                  # Infrastructure (CLIP, BLIP, Ollama clients)
└── schemas/                # Pydantic models
```

## API Documentation

### Chat Inference

**Endpoint**: `POST /api/v1/inference/chat`

**Headers**:

- `Content-Type`: `application/json`
- `X-Internal-Token`: `<YOUR_SECRET_TOKEN>`

**Request Body**:

```json
{
  "message": "What is this dish?",
  "image_url": "https://example.com/pho-bo.jpg",
  "history": []
}
```

| Field       | Type   | Description                                   |
|:------------|:-------|:----------------------------------------------|
| `message`   | string | User's text query.                            |
| `image_url` | string | (Optional) URL of the image to analyze.       |
| `history`   | list   | (Optional) Previous chat history for context. |

**Response**:

```json
{
  "response": "This is Pho Bo, a traditional Vietnamese beef noodle soup...",
  "intent": "food_analysis",
  "language": "vi"
}
```

## Installation & Setup

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/) running locally or remotely.

### 1. Clone the repository

```bash
git clone <repo_url>
cd DACN2_AIserver
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configuration

Create a `.env` file in the root directory:

```env
API_V1_STR=/api/v1
PROJECT_NAME="DACN2 AI Server"
INTERNAL_TOKEN=""
OLLAMA_BASE_URL=http://localhost:11434
ARTIFACTS_DIR=./artifacts
```

### 5. Run the Server

```bash
uvicorn serve.app.main:app --reload --host 0.0.0.0 --port 8000
```

## Docker Setup

This project can be started with Docker Compose on **Windows** and **macOS** (Docker Desktop).
It runs:

- `aiserver` (FastAPI + Uvicorn)
- `ollama` (LLM backend)

### Prerequisites

- Docker Desktop installed (Windows/macOS)
- `docker compose` available
- Ensure `./artifacts/` exists and contains:
    - `best.pt`
    - `food101_classes.json`
    - `model_config.json`

### 1) Build & start

From the repository root (where `docker-compose.yml` is located):

```bash
docker compose up --build
```

This will start:

- AIserver: http://localhost:8000
- Ollama: http://localhost:11434

To stop:

- Press `Ctrl + C`, then (optional) run:

```bash
docker compose down
```

### 2) Pull Ollama model (first-time only)

In a **new terminal tab/window** (keep `docker compose up` running), pull the model:

```bash
docker compose exec ollama ollama pull llama3.2:3b
```

Verify installed models:

```bash
docker compose exec ollama ollama list
```

### 3) Environment variables

The compose file provides sane defaults:

- `DEVICE=auto` (auto-select device; on Docker Desktop Win/Mac it usually runs on CPU)
- `OLLAMA_BASE_URL=http://ollama:11434` (container-to-container)
- `OLLAMA_MODEL=llama3.2:3b`
- `ENV=prod` (enables internal token check)

> When `ENV=prod`, requests to the inference API must include the `X-Internal-Token` header.

> **Note**: If you are running Ollama on your local machine (host), you may need to set
`OLLAMA_BASE_URL=http://host.docker.internal:11434` in your `.env` file so the container can access it.

## Testing

You can test the API using `curl`:

```bash
curl -X POST "http://localhost:8000/api/v1/inference/chat" \
     -H "Content-Type: application/json" \
     -H "X-Internal-Token: mysecrettoken123" \
     -d '{
           "message": "Phân tích hình ảnh này",
           "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Pho_Bo_-_Beef_Noodle_Soup.jpg/640px-Pho_Bo_-_Beef_Noodle_Soup.jpg"
         }'
```