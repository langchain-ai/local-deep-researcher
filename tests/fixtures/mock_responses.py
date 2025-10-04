"""Mock data for testing."""

MOCK_SUCCESSFUL_GRAPH_OUTPUT = {
    "success": True,
    "running_summary": "This is a comprehensive research summary about quantum computing. "
    "Quantum computers leverage quantum mechanical phenomena like superposition and entanglement "
    "to process information in fundamentally different ways than classical computers. "
    "Recent advances have demonstrated quantum supremacy in specific tasks, though practical "
    "applications remain limited by noise and decoherence issues. "
    "Major tech companies and research institutions are racing to develop more stable qubits "
    "and error correction schemes to enable scalable quantum computing.",
    "sources": [
        "https://example.com/quantum-computing-basics",
        "https://example.com/quantum-supremacy-2024",
        "https://example.com/qubit-stability-research",
    ],
    "error_message": None,
}

MOCK_FAILED_GRAPH_OUTPUT = {
    "success": False,
    "running_summary": None,
    "sources": [],
    "error_message": "Failed to generate summary",
}

MOCK_TIMEOUT_GRAPH_OUTPUT = {
    "success": False,
    "running_summary": None,
    "sources": [],
    "error_message": "Research request exceeded 5-minute timeout",
}

MOCK_SOURCES_GATHERED = [
    "https://example.com/article1\nhttps://example.com/article2",
    "Source: https://example.com/article3\nAnother source: https://example.com/article4",
    "https://example.com/article5",
]

MOCK_WEB_RESEARCH_RESULTS = [
    {
        "url": "https://example.com/article1",
        "content": "Sample content about the research topic...",
        "title": "Article 1 Title",
    },
    {
        "url": "https://example.com/article2",
        "content": "More detailed information about the subject...",
        "title": "Article 2 Title",
    },
]

MOCK_OLLAMA_CONFIG = {
    "configurable": {
        "local_llm": "llama3.2",
        "ollama_base_url": "http://localhost:11434/",
        "max_web_research_loops": 3,
        "search_api": "duckduckgo",
        "fetch_full_page": True,
        "strip_thinking_tokens": True,
        "use_tool_calling": False,
    }
}

MOCK_SUMMARY_STATE_DICT = {
    "research_topic": "Latest developments in quantum computing",
    "search_query": "quantum computing advances 2024",
    "web_research_results": [
        "Result 1: Quantum supremacy achieved...",
        "Result 2: New qubit designs improve stability...",
    ],
    "sources_gathered": MOCK_SOURCES_GATHERED,
    "research_loop_count": 2,
    "running_summary": "Quantum computing has made significant progress in recent years. "
    * 20,
}
