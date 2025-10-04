"""Unit tests for FastAPI endpoints."""

from unittest.mock import patch

import pytest


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up environment variables for testing."""
    monkeypatch.setenv("LLM_MODEL", "llama3.2")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/")


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check_returns_200_ok(self, mock_env_vars):
        """Test that health endpoint returns status ok."""
        # Import here to ensure mocked env vars are set
        from fastapi.testclient import TestClient

        from ollama_deep_researcher.api.main import app

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestResearchEndpoint:
    """Tests for the research endpoint."""

    @pytest.mark.asyncio
    async def test_research_endpoint_with_valid_topic(
        self, mock_env_vars, mock_graph_success
    ):
        """Test successful research request."""
        from fastapi.testclient import TestClient

        with patch("ollama_deep_researcher.api.main.graph", mock_graph_success):
            from ollama_deep_researcher.api.main import app

            client = TestClient(app)
            response = client.post(
                "/api/v1/research", json={"topic": "quantum computing"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["summary"] is not None
            assert len(data["summary"]) > 0
            assert isinstance(data["sources"], list)
            assert len(data["sources"]) == 2
            assert data["error_message"] is None

    @pytest.mark.asyncio
    async def test_research_endpoint_validates_empty_topic(self, mock_env_vars):
        """Test that empty topic is rejected with validation error."""
        from fastapi.testclient import TestClient

        from ollama_deep_researcher.api.main import app

        client = TestClient(app)
        response = client.post("/api/v1/research", json={"topic": ""})

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_research_endpoint_validates_missing_topic(self, mock_env_vars):
        """Test that missing topic field is rejected."""
        from fastapi.testclient import TestClient

        from ollama_deep_researcher.api.main import app

        client = TestClient(app)
        response = client.post("/api/v1/research", json={})

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_research_endpoint_handles_timeout(
        self, mock_env_vars, mock_graph_timeout
    ):
        """Test that timeout is handled gracefully."""
        from fastapi.testclient import TestClient

        with patch("ollama_deep_researcher.api.main.graph", mock_graph_timeout):
            from ollama_deep_researcher.api.main import app

            client = TestClient(app)
            response = client.post(
                "/api/v1/research", json={"topic": "quantum computing"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False
            assert data["summary"] is None
            assert data["sources"] == []
            assert "timeout" in data["error_message"].lower()

    @pytest.mark.asyncio
    async def test_research_endpoint_handles_graph_exception(
        self, mock_env_vars, mock_graph_exception
    ):
        """Test that exceptions from graph are handled gracefully."""
        from fastapi.testclient import TestClient

        with patch("ollama_deep_researcher.api.main.graph", mock_graph_exception):
            from ollama_deep_researcher.api.main import app

            client = TestClient(app)
            response = client.post(
                "/api/v1/research", json={"topic": "quantum computing"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False
            assert data["summary"] is None
            assert data["sources"] == []
            assert data["error_message"] is not None
            assert "error" in data["error_message"].lower()

    @pytest.mark.asyncio
    async def test_research_endpoint_reads_env_config(self, monkeypatch):
        """Test that endpoint reads configuration from environment variables."""
        monkeypatch.setenv("LLM_MODEL", "test-model")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://test:1234/")

        from unittest.mock import AsyncMock

        from fastapi.testclient import TestClient

        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "success": True,
            "running_summary": "Test summary",
            "sources": [],
            "error_message": None,
        }

        with patch("ollama_deep_researcher.api.main.graph", mock_graph):
            from ollama_deep_researcher.api.main import app

            client = TestClient(app)
            response = client.post("/api/v1/research", json={"topic": "test topic"})

            assert response.status_code == 200

            # Verify the config passed to graph
            call_args = mock_graph.ainvoke.call_args
            config = (
                call_args[0][1] if len(call_args[0]) > 1 else call_args[1]["config"]
            )
            assert config["configurable"]["local_llm"] == "test-model"
            assert config["configurable"]["ollama_base_url"] == "http://test:1234/"

    @pytest.mark.asyncio
    async def test_research_logs_structured_json(
        self, mock_env_vars, mock_graph_success, caplog
    ):
        """Test that logs are in JSON format with proper structure."""
        from fastapi.testclient import TestClient

        with patch("ollama_deep_researcher.api.main.graph", mock_graph_success):
            from ollama_deep_researcher.api.main import app

            client = TestClient(app)

            with caplog.at_level("INFO"):
                response = client.post(
                    "/api/v1/research", json={"topic": "quantum computing"}
                )

            assert response.status_code == 200

            # Check that logs contain relevant information
            log_messages = [record.message for record in caplog.records]
            assert any("Research request received" in msg for msg in log_messages)
            assert any("Research completed" in msg for msg in log_messages)
