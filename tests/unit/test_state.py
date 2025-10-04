"""Unit tests for state management."""

from dataclasses import asdict

from ollama_deep_researcher.state import (
    SummaryState,
    SummaryStateInput,
    SummaryStateOutput,
)


class TestSummaryStateOutput:
    """Tests for the enhanced SummaryStateOutput dataclass."""

    def test_summary_state_output_defaults(self):
        """Test that SummaryStateOutput has correct default values."""
        output = SummaryStateOutput()

        assert output.success is False
        assert output.running_summary is None
        assert output.sources == []
        assert output.error_message is None

    def test_summary_state_output_success_case(self):
        """Test creating a successful output state."""
        summary = "This is a comprehensive research summary " * 20
        sources = [
            "https://example.com/source1",
            "https://example.com/source2",
            "https://example.com/source3",
        ]

        output = SummaryStateOutput(
            success=True, running_summary=summary, sources=sources, error_message=None
        )

        assert output.success is True
        assert output.running_summary == summary
        assert output.sources == sources
        assert output.error_message is None

    def test_summary_state_output_error_case(self):
        """Test creating a failed output state with error message."""
        error_msg = "Failed to generate summary"

        output = SummaryStateOutput(
            success=False, running_summary=None, sources=[], error_message=error_msg
        )

        assert output.success is False
        assert output.running_summary is None
        assert output.sources == []
        assert output.error_message == error_msg

    def test_summary_state_output_serialization(self):
        """Test that SummaryStateOutput can be serialized to dict."""
        output = SummaryStateOutput(
            success=True,
            running_summary="Test summary",
            sources=["https://example.com"],
            error_message=None,
        )

        output_dict = asdict(output)

        assert isinstance(output_dict, dict)
        assert output_dict["success"] is True
        assert output_dict["running_summary"] == "Test summary"
        assert output_dict["sources"] == ["https://example.com"]
        assert output_dict["error_message"] is None

    def test_summary_state_output_partial_failure(self):
        """Test output state when summary exists but no sources found."""
        output = SummaryStateOutput(
            success=False,
            running_summary="Partial summary",
            sources=[],
            error_message="No sources found",
        )

        assert output.success is False
        assert output.running_summary == "Partial summary"
        assert len(output.sources) == 0
        assert "sources" in output.error_message.lower()


class TestSummaryStateInput:
    """Tests for SummaryStateInput dataclass."""

    def test_summary_state_input_creation(self):
        """Test creating a SummaryStateInput."""
        input_state = SummaryStateInput(research_topic="Test topic")

        assert input_state.research_topic == "Test topic"

    def test_summary_state_input_default(self):
        """Test default value of SummaryStateInput."""
        input_state = SummaryStateInput()

        assert input_state.research_topic is None


class TestSummaryState:
    """Tests for the main SummaryState dataclass."""

    def test_summary_state_creation(self):
        """Test creating a SummaryState with all fields."""
        state = SummaryState(
            research_topic="Quantum computing",
            search_query="quantum computing basics",
            web_research_results=["result1", "result2"],
            sources_gathered=["source1", "source2"],
            research_loop_count=2,
            running_summary="Initial summary",
        )

        assert state.research_topic == "Quantum computing"
        assert state.search_query == "quantum computing basics"
        assert len(state.web_research_results) == 2
        assert len(state.sources_gathered) == 2
        assert state.research_loop_count == 2
        assert state.running_summary == "Initial summary"

    def test_summary_state_defaults(self):
        """Test default values of SummaryState."""
        state = SummaryState()

        assert state.research_topic is None
        assert state.search_query is None
        assert state.web_research_results == []
        assert state.sources_gathered == []
        assert state.research_loop_count == 0
        assert state.running_summary is None

    def test_summary_state_accumulation(self):
        """Test that annotated lists accumulate properly."""
        state = SummaryState(
            web_research_results=["result1"],
            sources_gathered=["source1"],
        )

        # Simulate accumulation (this is handled by LangGraph in practice)
        assert "result1" in state.web_research_results
        assert "source1" in state.sources_gathered
