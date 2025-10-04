FROM --platform=$BUILDPLATFORM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    python3-dev \
    libssl-dev \
    libffi-dev \
    rustc \
    cargo \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager (use pip for safer cross-arch install)
RUN pip install uv
ENV PATH="/root/.local/bin:${PATH}"

# Copy the repository content
COPY . /app

# Install dependencies using uv
RUN uv sync

# Provide default environment variables
ENV OLLAMA_BASE_URL="http://localhost:11434/"
ENV LLM_MODEL="llama3.2"

# Expose the port for FastAPI service
EXPOSE 8000

# Launch the FastAPI application with uvicorn
CMD ["uv", "run", "uvicorn", "ollama_deep_researcher.api.main:app", "--host", "0.0.0.0", "--port", "8000"]