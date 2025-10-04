"""FastAPI application entry point."""

import asyncio
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from ollama_deep_researcher.api.logger import logger
from ollama_deep_researcher.api.models import (
    HealthResponse,
    ResearchRequest,
    ResearchResponse,
)
from ollama_deep_researcher.graph import graph

# Load environment variables from .env file
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app."""
    logger.info("Starting ollama-deep-researcher API service")
    yield
    logger.info("Shutting down ollama-deep-researcher API service")


app = FastAPI(
    title="Ollama Deep Researcher API",
    description="Production-ready research agent API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for load balancers."""
    return HealthResponse(status="ok")


@app.post("/api/v1/research", response_model=ResearchResponse)
async def run_research(request: ResearchRequest):
    """Execute deep research on a given topic."""
    logger.info("Research request received", extra={"topic": request.topic})

    try:
        # Build configuration from environment
        config = {
            "configurable": {
                "local_llm": os.getenv("LLM_MODEL", "llama3.2"),
                "ollama_base_url": os.getenv(
                    "OLLAMA_BASE_URL", "http://localhost:11434/"
                ),
            }
        }

        # Execute graph with timeout
        result = await asyncio.wait_for(
            graph.ainvoke({"research_topic": request.topic}, config=config),
            timeout=300.0,  # 5-minute timeout
        )

        # Map graph output to API response
        response = ResearchResponse(
            success=result.get("success", False),
            summary=result.get("running_summary"),
            sources=result.get("sources", []),
            error_message=result.get("error_message"),
        )

        logger.info(
            "Research completed",
            extra={
                "topic": request.topic,
                "success": response.success,
                "source_count": len(response.sources),
            },
        )

        return response

    except asyncio.TimeoutError:
        logger.error("Research timeout", extra={"topic": request.topic})
        return ResearchResponse(
            success=False,
            summary=None,
            sources=[],
            error_message="Research request exceeded 5-minute timeout",
        )
    except Exception as e:
        logger.error("Research failed", extra={"topic": request.topic, "error": str(e)})
        return ResearchResponse(
            success=False,
            summary=None,
            sources=[],
            error_message=f"Internal error: {str(e)}",
        )
