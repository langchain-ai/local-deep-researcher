from ollama_deep_researcher.graph import route_research
from ollama_deep_researcher.state import SummaryState


def test_route_research_continues_before_max_loops():
    state = SummaryState(research_topic="langgraph", research_loop_count=2)
    config = {"configurable": {"max_web_research_loops": 3}}

    assert route_research(state, config) == "web_research"


def test_route_research_finalizes_at_max_loops():
    state = SummaryState(research_topic="langgraph", research_loop_count=3)
    config = {"configurable": {"max_web_research_loops": 3}}

    assert route_research(state, config) == "finalize_summary"
