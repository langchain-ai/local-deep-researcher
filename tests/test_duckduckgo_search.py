import unittest
from unittest.mock import patch

from ollama_deep_researcher.configuration import Configuration
from ollama_deep_researcher.graph import web_research
from ollama_deep_researcher.state import SummaryState
from ollama_deep_researcher.utils import duckduckgo_search


class DuckDuckGoSearchTests(unittest.TestCase):
    @patch("ollama_deep_researcher.utils.DDGS")
    def test_uses_duckduckgo_backend_and_configured_region(self, mock_ddgs):
        mock_ddgs.return_value.text.return_value = [
            {
                "title": "Example result",
                "href": "https://example.com/result",
                "body": "Example content",
            }
        ]

        result = duckduckgo_search(
            "example query",
            max_results=2,
            region="uk-en",
        )

        mock_ddgs.return_value.text.assert_called_once_with(
            "example query",
            region="uk-en",
            max_results=2,
            backend="duckduckgo",
        )
        self.assertEqual(
            result,
            {
                "results": [
                    {
                        "title": "Example result",
                        "url": "https://example.com/result",
                        "content": "Example content",
                        "raw_content": "Example content",
                    }
                ]
            },
        )

    def test_region_can_be_set_from_runnable_config(self):
        configuration = Configuration.from_runnable_config(
            {"configurable": {"duckduckgo_region": "de-de"}}
        )

        self.assertEqual(configuration.duckduckgo_region, "de-de")

    @patch("ollama_deep_researcher.graph.format_sources", return_value="")
    @patch("ollama_deep_researcher.graph.deduplicate_and_format_sources", return_value="")
    @patch(
        "ollama_deep_researcher.graph.duckduckgo_search",
        return_value={"results": []},
    )
    def test_web_research_forwards_configured_region(
        self,
        mock_search,
        _mock_format_results,
        _mock_format_sources,
    ):
        state = SummaryState(search_query="example query")

        web_research(
            state,
            {
                "configurable": {
                    "search_api": "duckduckgo",
                    "duckduckgo_region": "de-de",
                    "fetch_full_page": False,
                }
            },
        )

        mock_search.assert_called_once_with(
            "example query",
            max_results=3,
            fetch_full_page=False,
            region="de-de",
        )


if __name__ == "__main__":
    unittest.main()
