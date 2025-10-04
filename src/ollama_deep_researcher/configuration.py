import os
from enum import Enum
from typing import Any, Literal, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"
    SEARXNG = "searxng"


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
    search_api: Literal["perplexity", "tavily", "duckduckgo", "searxng"] = Field(
        default="duckduckgo", title="Search API", description="Web search API to use"
    )
    fetch_full_page: bool = Field(
        default=True,
        title="Fetch Full Page",
        description="Include the full page content in the search results",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434/",
        title="Ollama Base URL",
        description="Base URL for Ollama API",
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

        # Environment variable name mappings (env_var_name -> field_name)
        env_mappings = {
            "LLM_MODEL": "local_llm",
            "OLLAMA_BASE_URL": "ollama_base_url",
        }

        # Get raw values from environment or config
        # Priority: configurable > environment > defaults
        raw_values: dict[str, Any] = {}
        for name in cls.model_fields.keys():
            # Check if value is in configurable first (highest priority)
            if name in configurable:
                raw_values[name] = configurable[name]
                continue

            # Check custom env mapping
            env_name = None
            for env_var, field_name in env_mappings.items():
                if field_name == name:
                    env_name = env_var
                    break

            # Get from environment (use custom name if available)
            if env_name and env_name in os.environ:
                raw_values[name] = os.environ[env_name]
            elif name.upper() in os.environ:
                raw_values[name] = os.environ[name.upper()]
            else:
                raw_values[name] = None

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)
