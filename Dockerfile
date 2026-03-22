FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    "datasets>=4.0.0" \
    "huggingface_hub>=0.20.0" \
    "openai>=1.0.0" \
    "python-dotenv>=1.0.0" \
    "fastapi>=0.115.0" \
    "uvicorn>=0.32.0" \
    "sse-starlette>=2.0.0"

# Copy app
COPY . .

# HF Spaces expects port 7860
EXPOSE 7860

CMD ["python", "web/app.py"]
