import os
from enum import Enum
from pydantic import BaseModel, Field
from typing import Any, Optional, Literal

from langchain_core.runnables import RunnableConfig


class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"
    SEARXNG = "searxng"


# Default base URLs per serving engine. Used only when llm_base_url is not
# explicitly set. Ollama uses its native API (no /v1 suffix); the others are
# reached through their OpenAI-compatible /v1 endpoint.
DEFAULT_LLM_BASE_URLS: dict[str, str] = {
    "ollama": "http://localhost:11434/",
    "lmstudio": "http://localhost:1234/v1",
    "vllm": "http://localhost:8000/v1",
    "sglang": "http://localhost:30000/v1",
}


class Configuration(BaseModel):
    """The configurable fields for the research assistant."""

    max_web_research_loops: int = Field(
        default=3,
        title="Research Depth",
        description="Number of research iterations to perform",
    )
    local_llm: str = Field(
        default="llama3.2",
        title="LLM Model Name",
        description="Name of the LLM model to use",
    )
    llm_provider: Literal["ollama", "lmstudio", "vllm", "sglang"] = Field(
        default="ollama",
        title="LLM Provider",
        description=(
            "Serving engine for the LLM. All are reached through the same "
            "OpenAI-compatible /v1 API; this only selects the default base URL "
            "when llm_base_url is not set."
        ),
    )
    search_api: Literal["perplexity", "tavily", "duckduckgo", "searxng"] = Field(
        default="duckduckgo", title="Search API", description="Web search API to use"
    )
    fetch_full_page: bool = Field(
        default=True,
        title="Fetch Full Page",
        description="Include the full page content in the search results",
    )
    llm_base_url: Optional[str] = Field(
        default=None,
        title="LLM Base URL",
        description=(
            "Base URL for the OpenAI-compatible /v1 API. If unset, a default is "
            "chosen from llm_provider (see DEFAULT_LLM_BASE_URLS)."
        ),
    )
    llm_api_key: str = Field(
        default="not-needed-for-local-models",
        title="LLM API Key",
        description="API key for the LLM server (only needed if it was started with an API key)",
    )
    strip_thinking_tokens: bool = Field(
        default=True,
        title="Strip Thinking Tokens",
        description="Whether to strip <think> tokens from model responses",
    )
    use_tool_calling: bool = Field(
        default=False,
        title="Use Tool Calling",
        description="Use tool calling instead of JSON mode for structured output",
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)

    def resolve_llm_base_url(self) -> str:
        """Return the base URL to use, falling back to the provider default."""
        return self.llm_base_url or DEFAULT_LLM_BASE_URLS[self.llm_provider]
