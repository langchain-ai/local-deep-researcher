"""LMStudio integration for the research assistant."""

import json
import logging
from typing import Any, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.messages import (
    BaseMessage,
)
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI
from pydantic import Field

# Set up logging
logger = logging.getLogger(__name__)


class ChatLMStudio(ChatOpenAI):
    """Chat model that uses LMStudio's OpenAI-compatible API."""

    format: Optional[str] = Field(
        default=None, description="Format for the response (e.g., 'json')"
    )

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: str = "qwen_qwq-32b",
        temperature: float = 0.7,
        format: Optional[str] = None,
        api_key: str = "not-needed-for-local-models",
        **kwargs: Any,
    ):
        """Initialize the ChatLMStudio.

        Args:
            base_url: Base URL for LMStudio's OpenAI-compatible API
            model: Model name to use
            temperature: Temperature for sampling
            format: Format for the response (e.g., "json")
            api_key: API key (not actually used, but required by OpenAI client)
            **kwargs: Additional arguments to pass to the OpenAI client
        """
        # Initialize the base class
        super().__init__(
            base_url=base_url,
            model=model,
            temperature=temperature,
            api_key=api_key,
            **kwargs,
        )
        self.format = format

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response using LMStudio's OpenAI-compatible API."""

        if self.format == "json":
            # Set response_format for JSON mode
            kwargs["response_format"] = {"type": "json_object"}
            logger.info(f"Using response_format={kwargs['response_format']}")

        # Call the parent class's _generate method
        result = super()._generate(messages, stop, run_manager, **kwargs)

        # If JSON format is requested, try to clean up the response
        if self.format == "json" and result.generations:
            try:
                # Get the raw text
                raw_text = result.generations[0][0].text
                logger.info(f"Raw model response: {raw_text}")

                # Try to find JSON in the response
                json_start = raw_text.find("{")
                json_end = raw_text.rfind("}") + 1

                if json_start >= 0 and json_end > json_start:
                    # Extract just the JSON part
                    json_text = raw_text[json_start:json_end]
                    # Validate it's proper JSON
                    json.loads(json_text)
                    logger.info(f"Cleaned JSON: {json_text}")
                    # Update the generation with the cleaned JSON
                    result.generations[0][0].text = json_text
                else:
                    logger.warning("Could not find JSON in response")
            except Exception as e:
                logger.error(f"Error processing JSON response: {str(e)}")
                # If any error occurs during cleanup, just use the original response
                pass

        return result
