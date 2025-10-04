"""Unit tests for configuration management."""

from ollama_deep_researcher.configuration import Configuration


class TestConfiguration:
    """Tests for the Configuration class."""

    def test_configuration_reads_env_variables(self, monkeypatch):
        """Test that Configuration reads from environment variables."""
        monkeypatch.setenv("LLM_MODEL", "test-model")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://test:1234/")

        config = Configuration.from_runnable_config({"configurable": {}})

        assert config.local_llm == "test-model"
        assert config.ollama_base_url == "http://test:1234/"

    def test_configuration_defaults_when_no_env(self, monkeypatch):
        """Test default values when environment variables are not set."""
        # Clear any existing env vars
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

        config = Configuration()

        assert config.local_llm == "llama3.2"
        assert config.ollama_base_url == "http://localhost:11434/"

    def test_configuration_from_runnable_config(self):
        """Test creating Configuration from RunnableConfig."""
        runnable_config = {
            "configurable": {
                "local_llm": "custom-model",
                "ollama_base_url": "http://custom:9999/",
                "max_web_research_loops": 5,
            }
        }

        config = Configuration.from_runnable_config(runnable_config)

        assert config.local_llm == "custom-model"
        assert config.ollama_base_url == "http://custom:9999/"
        assert config.max_web_research_loops == 5

    def test_configuration_prioritizes_config_over_env(self, monkeypatch):
        """Test that explicit config values override environment variables."""
        monkeypatch.setenv("LLM_MODEL", "env-model")

        runnable_config = {
            "configurable": {
                "local_llm": "config-model",
            }
        }

        config = Configuration.from_runnable_config(runnable_config)

        # Config value should take precedence
        assert config.local_llm == "config-model"

    def test_lmstudio_fields_removed(self):
        """Test that LM Studio related fields are not present in Configuration."""
        config = Configuration()

        # These fields should not exist
        assert not hasattr(config, "llm_provider")
        assert not hasattr(config, "lmstudio_base_url")

    def test_configuration_has_ollama_fields(self):
        """Test that Ollama-specific fields are present."""
        config = Configuration()

        assert hasattr(config, "local_llm")
        assert hasattr(config, "ollama_base_url")
        assert isinstance(config.local_llm, str)
        assert isinstance(config.ollama_base_url, str)

    def test_configuration_all_fields_have_defaults(self):
        """Test that all Configuration fields have sensible defaults."""
        config = Configuration()

        # Core fields
        assert config.local_llm is not None
        assert config.ollama_base_url is not None
        assert config.max_web_research_loops > 0

        # Feature flags
        assert isinstance(config.strip_thinking_tokens, bool)
        assert isinstance(config.use_tool_calling, bool)
        assert isinstance(config.fetch_full_page, bool)

        # Search API
        assert config.search_api is not None

    def test_configuration_validates_search_api(self):
        """Test that search_api accepts valid values."""
        valid_apis = ["perplexity", "tavily", "duckduckgo", "searxng"]

        for api in valid_apis:
            config = Configuration(search_api=api)
            assert config.search_api == api

    def test_configuration_env_override_for_all_configurable_fields(self, monkeypatch):
        """Test environment variable override for multiple fields."""
        monkeypatch.setenv("LLM_MODEL", "env-model")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://env:5555/")
        monkeypatch.setenv("MAX_WEB_RESEARCH_LOOPS", "7")

        config = Configuration.from_runnable_config({})

        # Note: from_runnable_config uses custom env mapping (LLM_MODEL -> local_llm)
        assert config.local_llm == "env-model"
        assert config.ollama_base_url == "http://env:5555/"

    def test_configuration_partial_config(self):
        """Test creating Configuration with only some fields specified."""
        config = Configuration(
            local_llm="partial-model",
            # Other fields should use defaults
        )

        assert config.local_llm == "partial-model"
        assert config.ollama_base_url == "http://localhost:11434/"  # default
        assert config.max_web_research_loops == 3  # default


class TestConfigurationIntegration:
    """Integration tests for Configuration in various scenarios."""

    def test_configuration_matches_api_expectations(self):
        """Test that Configuration structure matches what API expects."""
        config = Configuration(
            local_llm="llama3.2", ollama_base_url="http://localhost:11434/"
        )

        # These are the fields the API will use
        assert hasattr(config, "local_llm")
        assert hasattr(config, "ollama_base_url")

        # These fields should NOT exist (LM Studio removed)
        assert not hasattr(config, "lmstudio_base_url")

    def test_configuration_from_empty_runnable_config(self):
        """Test creating Configuration from empty RunnableConfig."""
        config = Configuration.from_runnable_config(None)

        # Should use all defaults
        assert config.local_llm == "llama3.2"
        assert config.ollama_base_url == "http://localhost:11434/"
