"""Unit tests for LangGraph node functions."""

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from ollama_deep_researcher.configuration import Configuration
from ollama_deep_researcher.graph import (
    finalize_summary,
    get_llm,
    summarize_sources,
    web_research,
)
from ollama_deep_researcher.state import SummaryState


class TestFinalizeSummary:
    """Tests for the finalize_summary node."""

    def test_finalize_summary_extracts_source_urls(self):
        """Test that finalize_summary extracts URLs from sources_gathered."""
        state = SummaryState(
            research_topic="Test topic",
            running_summary="Test summary with sufficient length " * 10,
            sources_gathered=[
                "https://example.com/article1\nhttps://example.com/article2",
                "https://example.com/article3",
                "Some text\nhttps://example.com/article4",
            ],
        )

        result = finalize_summary(state)

        assert "sources" in result
        assert len(result["sources"]) == 4
        assert "https://example.com/article1" in result["sources"]
        assert "https://example.com/article2" in result["sources"]
        assert "https://example.com/article3" in result["sources"]
        assert "https://example.com/article4" in result["sources"]

    def test_finalize_summary_success_with_valid_data(self):
        """Test success=True when summary and sources are present."""
        state = SummaryState(
            research_topic="Test topic",
            running_summary="This is a comprehensive summary with more than fifty characters in total.",
            sources_gathered=[
                "https://example.com/source1",
                "https://example.com/source2",
            ],
        )

        result = finalize_summary(state)

        assert result["success"] is True
        assert result["error_message"] is None
        assert len(result["sources"]) > 0

    def test_finalize_summary_fails_with_short_summary(self):
        """Test success=False when summary is too short."""
        state = SummaryState(
            research_topic="Test topic",
            running_summary="Short",  # Less than 50 chars
            sources_gathered=["https://example.com/source1"],
        )

        result = finalize_summary(state)

        assert result["success"] is False
        assert "summary" in result["error_message"].lower()

    def test_finalize_summary_fails_with_no_sources(self):
        """Test success=False when no sources are found."""
        state = SummaryState(
            research_topic="Test topic",
            running_summary="This is a comprehensive summary with more than fifty characters in total.",
            sources_gathered=[],  # No sources
        )

        result = finalize_summary(state)

        assert result["success"] is False
        assert "sources" in result["error_message"].lower()

    def test_finalize_summary_fails_with_no_summary(self):
        """Test success=False when summary is None."""
        state = SummaryState(
            research_topic="Test topic",
            running_summary=None,
            sources_gathered=["https://example.com/source1"],
        )

        result = finalize_summary(state)

        assert result["success"] is False
        assert result["error_message"] is not None

    def test_finalize_summary_deduplicates_sources(self):
        """Test that duplicate sources are removed."""
        state = SummaryState(
            research_topic="Test topic",
            running_summary="Test summary with sufficient length " * 10,
            sources_gathered=[
                "https://example.com/article1\nhttps://example.com/article1",
                "https://example.com/article1",
            ],
        )

        result = finalize_summary(state)

        # Should only have one instance of article1
        assert result["sources"].count("https://example.com/article1") == 1


class TestGetLlm:
    """Tests for the get_llm helper function."""

    def test_get_llm_returns_ollama_instance(self):
        """Test that get_llm returns ChatOllama instance."""
        config = Configuration(
            local_llm="llama3.2",
            ollama_base_url="http://localhost:11434/",
            use_tool_calling=False,
        )

        llm = get_llm(config)

        from langchain_ollama import ChatOllama

        assert isinstance(llm, ChatOllama)

    def test_get_llm_with_json_mode(self):
        """Test that get_llm configures JSON mode correctly."""
        config = Configuration(
            local_llm="llama3.2",
            ollama_base_url="http://localhost:11434/",
            use_tool_calling=False,
        )

        llm = get_llm(config)

        # Check that format is set to json
        assert hasattr(llm, "format")
        assert llm.format == "json"

    def test_get_llm_with_tool_calling(self):
        """Test that get_llm works with tool calling mode."""
        config = Configuration(
            local_llm="llama3.2",
            ollama_base_url="http://localhost:11434/",
            use_tool_calling=True,
        )

        llm = get_llm(config)

        from langchain_ollama import ChatOllama

        assert isinstance(llm, ChatOllama)
        # In tool calling mode, format should not be set to json
        assert not hasattr(llm, "format") or llm.format != "json"

    def test_get_llm_uses_correct_config_values(self):
        """Test that get_llm uses provided configuration values."""
        config = Configuration(
            local_llm="custom-model",
            ollama_base_url="http://custom:9999/",
            use_tool_calling=False,
        )

        llm = get_llm(config)

        assert llm.model == "custom-model"
        assert llm.base_url == "http://custom:9999/"


class TestWebResearch:
    """Tests for the web_research node with error handling."""

    @patch("ollama_deep_researcher.graph.duckduckgo_search")
    @patch("ollama_deep_researcher.graph.deduplicate_and_format_sources")
    @patch("ollama_deep_researcher.graph.format_sources")
    def test_web_research_success(
        self, mock_format, mock_dedupe, mock_search, mock_config
    ):
        """Test successful web research execution."""
        mock_search.return_value = [
            {"url": "https://example.com/1", "content": "Test content"}
        ]
        mock_dedupe.return_value = "Formatted search results"
        mock_format.return_value = "https://example.com/1"

        state = SummaryState(
            research_topic="Test topic",
            search_query="test query",
            research_loop_count=1,
        )

        result = web_research(state, mock_config)

        assert "sources_gathered" in result
        assert "research_loop_count" in result
        assert "web_research_results" in result
        assert result["research_loop_count"] == 2

    @patch("ollama_deep_researcher.graph.duckduckgo_search")
    def test_web_research_handles_search_exception(self, mock_search, mock_config):
        """Test that web_research handles search API exceptions."""
        mock_search.side_effect = Exception("Search API failed")

        state = SummaryState(
            research_topic="Test topic",
            search_query="test query",
            research_loop_count=1,
        )

        result = web_research(state, mock_config)

        # Should return error state instead of raising exception
        assert "sources_gathered" in result
        assert "research_loop_count" in result
        assert result["research_loop_count"] == 2
        # Error should be logged but execution continues
        assert (
            "Error" in result["sources_gathered"][0]
            or "Search failed" in result["web_research_results"][0]
        )


class TestSummarizeSources:
    """Tests for the summarize_sources node with error handling."""

    @patch("ollama_deep_researcher.graph.ChatOllama")
    def test_summarize_sources_success(self, mock_ollama_class, mock_config):
        """Test successful source summarization."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(
            content="This is a generated summary of the sources."
        )
        mock_ollama_class.return_value = mock_llm

        state = SummaryState(
            research_topic="Test topic",
            web_research_results=["Research result 1", "Research result 2"],
            running_summary=None,
        )

        result = summarize_sources(state, mock_config)

        assert "running_summary" in result
        assert result["running_summary"] is not None
        assert len(result["running_summary"]) > 0

    @patch("ollama_deep_researcher.graph.ChatOllama")
    def test_summarize_sources_updates_existing_summary(
        self, mock_ollama_class, mock_config
    ):
        """Test updating an existing summary with new context."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Updated summary")
        mock_ollama_class.return_value = mock_llm

        state = SummaryState(
            research_topic="Test topic",
            web_research_results=["Old result", "New result"],
            running_summary="Existing summary",
        )

        result = summarize_sources(state, mock_config)

        assert "running_summary" in result
        assert result["running_summary"] == "Updated summary"

    @patch("ollama_deep_researcher.graph.ChatOllama")
    def test_summarize_sources_handles_llm_exception(
        self, mock_ollama_class, mock_config
    ):
        """Test that summarize_sources handles LLM exceptions gracefully."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM failed")
        mock_ollama_class.return_value = mock_llm

        state = SummaryState(
            research_topic="Test topic",
            web_research_results=["Research result"],
            running_summary="Existing summary",
        )

        result = summarize_sources(state, mock_config)

        # Should return existing summary or fallback instead of raising
        assert "running_summary" in result
        assert result["running_summary"] is not None
