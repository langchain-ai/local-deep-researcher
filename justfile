# ==============================================================================
# justfile for FastAPI Project Automation
# ==============================================================================

set dotenv-load 

PROJECT_NAME := env("PROJECT_NAME", "fastapi-sandbox")

DEV_PROJECT_NAME := PROJECT_NAME + "-dev"
TEST_PROJECT_NAME := PROJECT_NAME + "-test"

# default target
default: help

# Show available recipes
help:
    @echo "Usage: just [recipe]"
    @echo "Available recipes:"
    @just --list | tail -n +2 | awk '{printf "  \033[36m%-20s\033[0m %s\n", $1, substr($0, index($0, $2))}'

# ==============================================================================
# Environment Setup
# ==============================================================================

# Initialize project: install dependencies, create .env file and pull required Docker images
setup:
    @echo "Installing python dependencies with uv..."
    @uv sync
    @echo "Creating environment file..."
    @if [ ! -f .env ] && [ -f .env.example ]; then \
        echo "Creating .env from .env.example..."; \
        cp .env.example .env; \
        echo "✅ Environment file created (.env)"; \
    else \
        echo ".env already exists. Skipping creation."; \
    fi
    @echo "💡 You can customize .env for your specific needs:"

# ==============================================================================
# Development Environment Commands
# ==============================================================================

# Start development server with uvicorn
dev:
  @uv run uvicorn api.main:app --host ${API_HOST_BIND_IP} --port ${API_HOST_PORT} --reload

# ==============================================================================
# CODE QUALITY
# ==============================================================================

# Format code using Black and fix issues with Ruff
format:
	@uv run black .
	@uv run ruff check . --fix

# Perform static code analysis using Black and Ruff
lint:
  @uv run black --check .
  @uv run ruff check .

# ==============================================================================
# TESTING
# ==============================================================================

# Run all tests
test: unit-test build-test
    @echo "✅ All tests passed!"

# Run unit tests locally (no external dependencies)
unit-test:
    @echo "🚀 Running unit tests (local)..."
    @uv run pytest tests/unit

# Run integration tests (requires Ollama)
intg-test:
    @echo "🚀 Running integration tests (requires Ollama)..."
    @uv run pytest tests/intg

# Build Docker image for testing without leaving artifacts
build-test:
    @echo "Building Docker image for testing..."
    @TEMP_IMAGE_TAG=$(date +%s)-build-test; \
    @docker build --tag temp-build-test:$TEMP_IMAGE_TAG -f api/Dockerfile . && \
    @echo "Build successful. Cleaning up temporary image..." && \
    @docker rmi temp-build-test:$TEMP_IMAGE_TAG || true

# ==============================================================================
# CLEANUP
# ==============================================================================

# Remove __pycache__ and .venv to make project lightweight
clean:
  @echo "🧹 Cleaning up project..."
  @find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
  @rm -rf .venv
  @rm -rf .pytest_cache
  @rm -rf .ruff_cache
  @rm -f test_db.sqlite3
  @echo "✅ Cleanup completed"