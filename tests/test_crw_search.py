"""Tests for the fastCRW search provider in ollama_deep_researcher.utils.

These tests mock the HTTP layer so no network access is required.
"""

from unittest.mock import MagicMock, patch

from ollama_deep_researcher.utils import crw_search


def _make_response(payload):
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


@patch("ollama_deep_researcher.utils.requests.post")
def test_crw_search_parses_results(mock_post):
    mock_post.return_value = _make_response(
        {
            "success": True,
            "data": [
                {
                    "title": "Example",
                    "url": "https://example.com",
                    "description": "An example description",
                }
            ],
        }
    )

    results = crw_search("python", max_results=3)

    assert results == {
        "results": [
            {
                "title": "Example",
                "url": "https://example.com",
                "content": "An example description",
                "raw_content": "An example description",
            }
        ]
    }

    # Default cloud endpoint and limit are forwarded to the API
    args, kwargs = mock_post.call_args
    assert args[0] == "https://fastcrw.com/api/v1/search"
    assert kwargs["json"] == {"query": "python", "limit": 3}


@patch.dict(
    "os.environ",
    {"CRW_API_KEY": "test-key", "CRW_BASE_URL": "http://localhost:3000/"},
    clear=False,
)
@patch("ollama_deep_researcher.utils.requests.post")
def test_crw_search_uses_env_overrides(mock_post):
    mock_post.return_value = _make_response({"success": True, "data": []})

    crw_search("python")

    args, kwargs = mock_post.call_args
    # Self-host base URL override is honored and trailing slash is stripped
    assert args[0] == "http://localhost:3000/v1/search"
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"


@patch("ollama_deep_researcher.utils.requests.post")
def test_crw_search_prefers_inline_markdown(mock_post):
    mock_post.return_value = _make_response(
        {
            "success": True,
            "data": [
                {
                    "title": "Example",
                    "url": "https://example.com",
                    "description": "An example description",
                    "markdown": "# Full markdown content",
                }
            ],
        }
    )

    results = crw_search("python", fetch_full_page=True)

    assert results["results"][0]["raw_content"] == "# Full markdown content"


@patch("ollama_deep_researcher.utils.requests.post")
def test_crw_search_handles_error_envelope(mock_post):
    mock_post.return_value = _make_response(
        {"success": False, "error": "rate limited"}
    )

    assert crw_search("python") == {"results": []}


@patch("ollama_deep_researcher.utils.requests.post")
def test_crw_search_skips_incomplete_results(mock_post):
    mock_post.return_value = _make_response(
        {
            "success": True,
            "data": [
                {"title": "No URL", "description": "missing url"},
                {
                    "title": "Complete",
                    "url": "https://example.com",
                    "description": "ok",
                },
            ],
        }
    )

    results = crw_search("python")

    assert len(results["results"]) == 1
    assert results["results"][0]["url"] == "https://example.com"
