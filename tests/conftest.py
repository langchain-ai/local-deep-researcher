import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_ollama import ChatOllama


# Prevent .env file from being loaded during tests
# Tests should explicitly set environment variables using monkeypatch
@pytest.fixture(scope="session", autouse=True)
def prevent_env_file_loading():
    """Ensure .env file doesn't interfere with tests."""
    # Mock load_dotenv to prevent .env from being loaded
    with patch("dotenv.load_dotenv"):
        # Clear any env vars that might have been loaded from .env
        env_vars_to_clear = [
            "LLM_MODEL",
            "OLLAMA_BASE_URL",
            "LOCAL_LLM",
            "LMSTUDIO_BASE_URL",
        ]
        for var in env_vars_to_clear:
            os.environ.pop(var, None)
        yield


@pytest.fixture
def mock_graph_success():
    """Mock successful graph execution."""
    mock = AsyncMock()
    mock.ainvoke.return_value = {
        "success": True,
        "running_summary": "This is a comprehensive research summary about the topic. "
        * 10,
        "sources": ["https://example.com/source1", "https://example.com/source2"],
        "error_message": None,
    }
    return mock


@pytest.fixture
def mock_graph_failure():
    """Mock failed graph execution."""
    mock = AsyncMock()
    mock.ainvoke.return_value = {
        "success": False,
        "running_summary": None,
        "sources": [],
        "error_message": "Failed to generate summary",
    }
    return mock


@pytest.fixture
def mock_graph_timeout():
    """Mock graph execution that times out."""
    mock = AsyncMock()
    mock.ainvoke.side_effect = asyncio.TimeoutError()
    return mock


@pytest.fixture
def mock_graph_exception():
    """Mock graph execution that raises exception."""
    mock = AsyncMock()
    mock.ainvoke.side_effect = Exception("Test exception")
    return mock


@pytest.fixture
def mock_ollama_llm():
    """Mock ChatOllama instance."""
    mock = MagicMock(spec=ChatOllama)
    mock.invoke.return_value = MagicMock(content="Test LLM response")
    return mock


@pytest.fixture
def mock_sources_gathered():
    """Mock sources gathered from web research."""
    return [
        "https://example.com/article1\nhttps://example.com/article2",
        "Source: https://example.com/article3",
    ]


@pytest.fixture
def sample_summary_state():
    """Sample SummaryState for testing."""
    from ollama_deep_researcher.state import SummaryState

    return SummaryState(
        research_topic="Test topic",
        search_query="test query",
        web_research_results=["result1", "result2"],
        sources_gathered=["https://example.com/1", "https://example.com/2"],
        research_loop_count=1,
        running_summary="Test summary with sufficient length " * 10,
    )


@pytest.fixture
def mock_config():
    """Mock RunnableConfig for testing."""
    return {
        "configurable": {
            "local_llm": "llama3.2",
            "ollama_base_url": "http://localhost:11434/",
            "max_web_research_loops": 3,
        }
    }
