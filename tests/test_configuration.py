import os
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from ollama_deep_researcher.configuration import Configuration


class ConfigurationTests(unittest.TestCase):
    def test_max_tokens_per_source_defaults_to_1000(self):
        configuration = Configuration()

        self.assertEqual(configuration.max_tokens_per_source, 1000)

    def test_max_tokens_per_source_can_be_configured(self):
        configuration = Configuration.from_runnable_config(
            {"configurable": {"max_tokens_per_source": 250}}
        )

        self.assertEqual(configuration.max_tokens_per_source, 250)

    @patch.dict(os.environ, {"MAX_TOKENS_PER_SOURCE": "500"})
    def test_environment_takes_precedence(self):
        configuration = Configuration.from_runnable_config(
            {"configurable": {"max_tokens_per_source": 250}}
        )

        self.assertEqual(configuration.max_tokens_per_source, 500)

    def test_max_tokens_per_source_must_be_positive(self):
        with self.assertRaises(ValidationError):
            Configuration(max_tokens_per_source=0)


if __name__ == "__main__":
    unittest.main()
