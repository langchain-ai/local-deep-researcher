#!/usr/bin/env python3
"""Multi-Agent Collaborative Research with AG2 + Local Deep Researcher.

Demonstrates how AG2 agents can use Local Deep Researcher's LangGraph
workflow as registered tools to perform multi-perspective research.
Three agents collaborate:

- Fact Checker: runs quick (1-iteration) searches on sub-questions
- Deep Researcher: runs deeper (3-iteration) searches on key aspects
- Report Writer: synthesizes all findings into a final summary

Each tool wraps the compiled LangGraph research graph from
``ollama_deep_researcher.graph`` with different iteration depths.

Requirements:
    pip install "ag2[openai]>=0.11.4,<1.0"
    pip install -e .  # install local-deep-researcher

    # Ollama must be running with a model pulled (default: llama3.2)
    ollama pull llama3.2

Usage:
    # Set your OpenAI API key (used by AG2 agents for orchestration)
    export OPENAI_API_KEY=your-key

    # Run the example
    python examples/ag2_multiagent_research.py

Note:
    The OPENAI_API_KEY is used only by AG2 for agent-to-agent orchestration.
    Local Deep Researcher uses Ollama (local LLM) and DuckDuckGo (no API key)
    by default. Configure via environment variables — see .env.example.
"""

import json
import os
from typing import Annotated

from autogen import (
    AssistantAgent,
    GroupChat,
    GroupChatManager,
    LLMConfig,
    UserProxyAgent,
)

from ollama_deep_researcher.graph import graph


def _run_research(topic: str, max_loops: int = 3) -> dict:
    """Invoke the LangGraph research workflow and return results.

    Args:
        topic: The research question or topic.
        max_loops: Number of search-summarize-reflect iterations.

    Returns:
        Dictionary with the research summary and sources.
    """
    config = {
        "configurable": {
            "max_web_research_loops": max_loops,
        }
    }
    result = graph.invoke({"research_topic": topic}, config=config)
    return {
        "topic": topic,
        "summary": result.get("running_summary", ""),
        "iterations": max_loops,
    }


# --------------------------------------------------------------------------- #
# Tool functions — wrap the LangGraph research graph for AG2 agents
# --------------------------------------------------------------------------- #
def quick_research(
    query: Annotated[str, "Research question to get a quick overview for"],
) -> str:
    """Run a quick 1-iteration research pass using Local Deep Researcher.

    Good for fact-checking, getting overviews, or answering sub-questions.
    Uses DuckDuckGo search with a single search-summarize cycle.
    """
    result = _run_research(query, max_loops=1)
    return json.dumps(result, indent=2)


def deep_research(
    query: Annotated[str, "Research question for in-depth iterative analysis"],
    iterations: Annotated[int, "Number of research iterations (1-5)"] = 3,
) -> str:
    """Run deep iterative research using Local Deep Researcher.

    Performs multiple search-summarize-reflect cycles, each refining the
    query based on knowledge gaps found in previous results. Returns a
    detailed summary with source citations.
    """
    result = _run_research(query, max_loops=min(max(iterations, 1), 5))
    return json.dumps(result, indent=2)


# --------------------------------------------------------------------------- #
# AG2 LLM configuration for agent orchestration
# --------------------------------------------------------------------------- #
llm_config = LLMConfig(
    {
        "model": "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "api_type": "openai",
    }
)

# --------------------------------------------------------------------------- #
# AG2 agents
# --------------------------------------------------------------------------- #
fact_checker = AssistantAgent(
    name="Fact_Checker",
    system_message=(
        "You are a fact checker. Your job is to gather quick overviews on "
        "specific sub-questions related to the main research topic. "
        "Use quick_research to check individual claims or get overviews "
        "of specific aspects. Break the main question into 2-3 focused "
        "sub-questions and research each one. Present your findings clearly."
    ),
    llm_config=llm_config,
)

deep_researcher = AssistantAgent(
    name="Deep_Researcher",
    system_message=(
        "You are a deep research analyst. After the Fact Checker has gathered "
        "initial findings, use deep_research to perform in-depth iterative "
        "analysis on the most important aspects. Focus on areas where the "
        "quick overviews revealed knowledge gaps or conflicting information. "
        "Use 3 iterations for thorough results."
    ),
    llm_config=llm_config,
)

report_writer = AssistantAgent(
    name="Report_Writer",
    system_message=(
        "You are a research report writer. After the Fact Checker and Deep "
        "Researcher have gathered findings, synthesize everything into a "
        "clear, well-structured summary with key findings, supporting "
        "evidence, and source citations. End your message with TERMINATE."
    ),
    llm_config=llm_config,
)

executor = UserProxyAgent(
    name="Executor",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config=False,
)

# Register tools — all agents can request them, executor runs them
for tool_fn, description in [
    (quick_research, "Run a quick 1-iteration research pass on a question"),
    (deep_research, "Run deep multi-iteration research on a question"),
]:
    executor.register_for_execution()(tool_fn)
    fact_checker.register_for_llm(description=description)(tool_fn)
    deep_researcher.register_for_llm(description=description)(tool_fn)
    report_writer.register_for_llm(description=description)(tool_fn)


# --------------------------------------------------------------------------- #
# Run the multi-agent research pipeline
# --------------------------------------------------------------------------- #
def main():
    """Run a collaborative multi-agent research session."""
    group_chat = GroupChat(
        agents=[executor, fact_checker, deep_researcher, report_writer],
        messages=[],
        max_round=15,
        speaker_selection_method="auto",
    )

    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=llm_config,
    )

    research_topic = (
        "What are the current approaches to making large language models "
        "more energy efficient, and what are the most promising techniques "
        "for reducing their computational costs?"
    )

    print(f"Starting multi-agent research on:\n{research_topic}\n")  # noqa: T201
    print("Pipeline: Fact Checker -> Deep Researcher -> Report Writer\n")  # noqa: T201

    executor.run(manager, message=research_topic).process()

    print("\n" + "=" * 60)  # noqa: T201
    print("Research complete!")  # noqa: T201
    print("=" * 60)  # noqa: T201


if __name__ == "__main__":
    main()
