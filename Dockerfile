FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git lsof && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
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

# Use exec form so Python receives SIGTERM directly and shuts down cleanly
STOPSIGNAL SIGTERM
CMD ["bash", "start.sh"]
