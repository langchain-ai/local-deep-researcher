"""Integration tests for the full API workflow.

These tests require Ollama to be running and are marked with @pytest.mark.integration.
Skip them in CI by running: pytest -m "not integration"
"""

import time

import pytest


@pytest.mark.integration
class TestFullResearchWorkflow:
    """Integration tests requiring actual Ollama service."""

    @pytest.mark.asyncio
    async def test_full_research_workflow_success(self):
        """Test complete research workflow with real Ollama instance."""
        pytest.skip("Requires Ollama to be running - run manually")

        from fastapi.testclient import TestClient

        from ollama_deep_researcher.api.main import app

        client = TestClient(app)

        # Make actual research request
        response = client.post(
            "/api/v1/research",
            json={"topic": "Python asyncio best practices"},
            timeout=120.0,  # Allow 2 minutes for real request
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "success" in data
        assert "summary" in data
        assert "sources" in data
        assert "error_message" in data

        # If successful, verify content quality
        if data["success"]:
            assert data["summary"] is not None
            assert len(data["summary"]) > 100
            assert len(data["sources"]) > 0
            assert (
                "python" in data["summary"].lower()
                or "asyncio" in data["summary"].lower()
            )

    @pytest.mark.asyncio
    async def test_research_with_ollama_unavailable(self, monkeypatch):
        """Test behavior when Ollama is not accessible."""
        pytest.skip("Manual test - requires stopping Ollama")

        # Point to non-existent Ollama instance
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:9999/")

        from fastapi.testclient import TestClient

        from ollama_deep_researcher.api.main import app

        client = TestClient(app)

        response = client.post(
            "/api/v1/research", json={"topic": "test topic"}, timeout=30.0
        )

        assert response.status_code == 200
        data = response.json()

        # Should return error response, not crash
        assert data["success"] is False
        assert data["error_message"] is not None

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self):
        """Test handling multiple concurrent research requests."""
        pytest.skip("Requires Ollama - run manually for stress testing")

        import asyncio

        from httpx import AsyncClient

        from ollama_deep_researcher.api.main import app

        async with AsyncClient(app=app, base_url="http://test") as client:
            topics = [
                "Machine learning basics",
                "Docker containers",
                "REST API design",
            ]

            # Send concurrent requests
            tasks = [
                client.post("/api/v1/research", json={"topic": topic}, timeout=120.0)
                for topic in topics
            ]

            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # All requests should complete
            assert len(responses) == 3

            # Check that all returned valid responses
            for response in responses:
                if not isinstance(response, Exception):
                    assert response.status_code == 200
                    data = response.json()
                    assert "success" in data


@pytest.mark.integration
class TestHealthCheckIntegration:
    """Integration test for health check endpoint."""

    def test_health_check_responds_quickly(self):
        """Test that health check has low latency."""
        from fastapi.testclient import TestClient

        from ollama_deep_researcher.api.main import app

        client = TestClient(app)

        start = time.time()
        response = client.get("/health")
        elapsed = time.time() - start

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        assert elapsed < 0.1  # Should respond in less than 100ms
