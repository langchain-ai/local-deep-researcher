"""vLLM integration for the research assistant.

vLLM exposes an OpenAI-compatible server (``vllm serve <model>``), so this is a
thin subclass of ``ChatOpenAI`` pointed at that endpoint. It is functionally
identical to the LMStudio integration; both talk the OpenAI ``/v1`` protocol.
The serving layer (and the hardware it runs on) is transparent to LangGraph.
"""

import json
import logging
from typing import Any, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI
from pydantic import Field

logger = logging.getLogger(__name__)


class ChatVLLM(ChatOpenAI):
    """Chat model that uses vLLM's OpenAI-compatible API."""

    format: Optional[str] = Field(
        default=None, description="Format for the response (e.g., 'json')"
    )

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        temperature: float = 0.7,
        format: Optional[str] = None,
        api_key: str = "not-needed-for-local-models",
        **kwargs: Any,
    ):
        """Initialize the ChatVLLM.

        Args:
            base_url: Base URL for vLLM's OpenAI-compatible API.
            model: Model name to use (must match what the server was launched with).
            temperature: Temperature for sampling.
            format: Format for the response (e.g., "json").
            api_key: API key. Only meaningful if vLLM was started with --api-key;
                a placeholder otherwise.
            **kwargs: Additional arguments passed to the OpenAI client.
        """
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
        """Generate a chat response using vLLM's OpenAI-compatible API."""

        if self.format == "json":
            # vLLM supports OpenAI-style JSON mode.
            kwargs["response_format"] = {"type": "json_object"}
            logger.info(f"Using response_format={kwargs['response_format']}")

        result = super()._generate(messages, stop, run_manager, **kwargs)

        # If JSON was requested, defensively extract the JSON object from the
        # response text (some models wrap it in prose). `ChatGeneration.text`
        # is a read-only view of `message.content`, so the message itself
        # must be updated for the cleanup to take effect downstream.
        if self.format == "json" and result.generations:
            try:
                generation = result.generations[0]
                raw_text = generation.text
                logger.info(f"Raw model response: {raw_text}")

                json_start = raw_text.find("{")
                json_end = raw_text.rfind("}") + 1

                if json_start >= 0 and json_end > json_start:
                    json_text = raw_text[json_start:json_end]
                    json.loads(json_text)  # validate
                    logger.info(f"Cleaned JSON: {json_text}")
                    generation.message.content = json_text
                else:
                    logger.warning("Could not find JSON in response")
            except Exception as e:
                logger.error(f"Error processing JSON response: {str(e)}")

        return result
