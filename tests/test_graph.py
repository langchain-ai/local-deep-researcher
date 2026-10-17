import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ollama_deep_researcher.graph import summarize_sources, web_research
from ollama_deep_researcher.state import SummaryState


class WebResearchTests(unittest.TestCase):
    @patch("ollama_deep_researcher.graph.deduplicate_and_format_sources")
    @patch("ollama_deep_researcher.graph.duckduckgo_search")
    def test_uses_configured_source_token_budget(self, search, format_sources):
        search_results = {"results": []}
        search.return_value = search_results
        format_sources.return_value = "Sources:"
        state = SummaryState(search_query="test query", research_loop_count=0)

        result = web_research(
            state,
            {
                "configurable": {
                    "search_api": "duckduckgo",
                    "fetch_full_page": False,
                    "max_tokens_per_source": 250,
                }
            },
        )

        format_sources.assert_called_once_with(
            search_results,
            max_tokens_per_source=250,
            fetch_full_page=False,
        )
        self.assertEqual(result["research_loop_count"], 1)


class SummarizeSourcesTests(unittest.TestCase):
    def setUp(self):
        self.state = SummaryState(
            research_topic="test topic",
            web_research_results=["new research"],
            running_summary="existing summary",
        )

    @patch("ollama_deep_researcher.graph.ChatOllama")
    def test_returns_trimmed_summary(self, chat_ollama):
        chat_ollama.return_value.invoke.return_value = SimpleNamespace(
            content="  updated summary  "
        )

        result = summarize_sources(self.state, {"configurable": {}})

        self.assertEqual(result, {"running_summary": "updated summary"})

    @patch("ollama_deep_researcher.graph.ChatOllama")
    def test_rejects_empty_summary(self, chat_ollama):
        chat_ollama.return_value.invoke.return_value = SimpleNamespace(content="   ")

        with self.assertRaisesRegex(ValueError, "empty summary"):
            summarize_sources(self.state, {"configurable": {}})

        self.assertEqual(self.state.running_summary, "existing summary")

    @patch("ollama_deep_researcher.graph.ChatOllama")
    def test_rejects_thinking_only_summary(self, chat_ollama):
        chat_ollama.return_value.invoke.return_value = SimpleNamespace(
            content="<think>reasoning</think>"
        )

        with self.assertRaisesRegex(ValueError, "empty summary"):
            summarize_sources(self.state, {"configurable": {}})

    @patch("ollama_deep_researcher.graph.ChatOllama")
    def test_rejects_non_text_summary(self, chat_ollama):
        chat_ollama.return_value.invoke.return_value = SimpleNamespace(content=[])

        with self.assertRaisesRegex(ValueError, "non-text summary content"):
            summarize_sources(self.state, {"configurable": {}})


if __name__ == "__main__":
    unittest.main()
