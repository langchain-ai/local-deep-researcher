"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    """Request model for research endpoint."""

    topic: str = Field(..., min_length=1, description="Research topic")


class ResearchResponse(BaseModel):
    """Response model for research endpoint."""

    success: bool = Field(..., description="Whether research completed successfully")
    summary: str | None = Field(None, description="Generated research summary")
    sources: list[str] = Field(default_factory=list, description="Source URLs")
    error_message: str | None = Field(None, description="Error details if failed")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(default="ok")
