"""Unit tests for the Exa search provider."""
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ollama_deep_researcher.utils import exa_search


def _make_response(results):
    return SimpleNamespace(results=results)


@patch("ollama_deep_researcher.utils.Exa")
def test_exa_search_uses_highlights_when_present(mock_exa_cls):
    client = MagicMock()
    client.headers = {}
    client.search_and_contents.return_value = _make_response(
        [
            SimpleNamespace(
                title="Result A",
                url="https://example.com/a",
                text="full body text",
                highlights=["first highlight", "second highlight"],
                summary=None,
            )
        ]
    )
    mock_exa_cls.return_value = client

    result = exa_search("test query", max_results=1, fetch_full_page=False)

    assert result["results"][0]["title"] == "Result A"
    assert result["results"][0]["url"] == "https://example.com/a"
    assert "first highlight" in result["results"][0]["content"]
    assert "second highlight" in result["results"][0]["content"]
    assert result["results"][0]["raw_content"] == result["results"][0]["content"]


@patch("ollama_deep_researcher.utils.Exa")
def test_exa_search_falls_back_to_summary_when_no_highlights(mock_exa_cls):
    client = MagicMock()
    client.headers = {}
    client.search_and_contents.return_value = _make_response(
        [
            SimpleNamespace(
                title="Result B",
                url="https://example.com/b",
                text="full body text",
                highlights=[],
                summary="A concise summary.",
            )
        ]
    )
    mock_exa_cls.return_value = client

    result = exa_search("test query", max_results=1, fetch_full_page=False)

    assert result["results"][0]["content"] == "A concise summary."


@patch("ollama_deep_researcher.utils.Exa")
def test_exa_search_falls_back_to_text_when_no_highlights_or_summary(mock_exa_cls):
    client = MagicMock()
    client.headers = {}
    client.search_and_contents.return_value = _make_response(
        [
            SimpleNamespace(
                title="Result C",
                url="https://example.com/c",
                text="just the body",
                highlights=None,
                summary=None,
            )
        ]
    )
    mock_exa_cls.return_value = client

    result = exa_search("test query", max_results=1, fetch_full_page=False)

    assert result["results"][0]["content"] == "just the body"


@patch("ollama_deep_researcher.utils.Exa")
def test_exa_search_fetch_full_page_returns_text_as_raw_content(mock_exa_cls):
    client = MagicMock()
    client.headers = {}
    client.search_and_contents.return_value = _make_response(
        [
            SimpleNamespace(
                title="Result D",
                url="https://example.com/d",
                text="the full page text",
                highlights=["a snippet"],
                summary=None,
            )
        ]
    )
    mock_exa_cls.return_value = client

    result = exa_search("test query", max_results=1, fetch_full_page=True)

    assert result["results"][0]["content"] == "a snippet"
    assert result["results"][0]["raw_content"] == "the full page text"


@patch("ollama_deep_researcher.utils.Exa")
def test_exa_search_skips_results_missing_required_fields(mock_exa_cls):
    client = MagicMock()
    client.headers = {}
    client.search_and_contents.return_value = _make_response(
        [
            SimpleNamespace(
                title="No URL",
                url="",
                text="body",
                highlights=["snippet"],
                summary=None,
            ),
            SimpleNamespace(
                title="Valid",
                url="https://example.com/ok",
                text="body",
                highlights=["snippet"],
                summary=None,
            ),
        ]
    )
    mock_exa_cls.return_value = client

    result = exa_search("test query", max_results=2)

    assert len(result["results"]) == 1
    assert result["results"][0]["url"] == "https://example.com/ok"


@patch("ollama_deep_researcher.utils.Exa")
def test_exa_search_sets_integration_header(mock_exa_cls):
    client = MagicMock()
    client.headers = {}
    client.search_and_contents.return_value = _make_response([])
    mock_exa_cls.return_value = client

    exa_search("test query")

    assert client.headers.get("x-exa-integration") == "local-deep-researcher"


@patch.dict(os.environ, {}, clear=True)
@patch("ollama_deep_researcher.utils.Exa")
def test_exa_search_passes_api_key_from_env(mock_exa_cls):
    os.environ["EXA_API_KEY"] = "test-key-123"
    client = MagicMock()
    client.headers = {}
    client.search_and_contents.return_value = _make_response([])
    mock_exa_cls.return_value = client

    exa_search("test query")

    mock_exa_cls.assert_called_once_with(api_key="test-key-123")


@patch("ollama_deep_researcher.utils.Exa")
def test_exa_search_forwards_max_results_and_type(mock_exa_cls):
    client = MagicMock()
    client.headers = {}
    client.search_and_contents.return_value = _make_response([])
    mock_exa_cls.return_value = client

    exa_search("a query", max_results=7)

    _, kwargs = client.search_and_contents.call_args
    assert kwargs["num_results"] == 7
    assert kwargs["type"] == "auto"
    assert "highlights" in kwargs["contents"]
