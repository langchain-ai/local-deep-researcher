# Test Suite for Ollama Deep Researcher

This directory contains comprehensive tests for the productionized FastAPI service.

## Test Structure

```
tests/
├── conftest.py                  # Shared pytest fixtures
├── fixtures/                    # Mock data and test fixtures
│   ├── __init__.py
│   └── mock_responses.py        # Centralized mock data
├── unit/                        # Unit tests (no external dependencies)
│   ├── __init__.py
│   ├── test_api.py             # FastAPI endpoint tests
│   ├── test_configuration.py   # Configuration management tests
│   ├── test_graph_nodes.py     # LangGraph node tests
│   └── test_state.py           # State dataclass tests
├── intg/                        # Integration tests (require Ollama)
│   ├── __init__.py
│   └── test_api_integration.py # Full workflow tests
└── README.md                    # This file
```

## Prerequisites

### Install Test Dependencies

```bash
# Using pip
pip install -e ".[test]"

# Using uv (recommended)
uv pip install -e ".[test]"
```

This installs:
- pytest >= 8.0.0
- pytest-asyncio >= 0.23.0
- pytest-mock >= 3.12.0
- pytest-cov >= 4.1.0

## Running Tests

### Run All Unit Tests (No External Dependencies)

```bash
# Using just
just unit-test

# Or directly with pytest
pytest tests/unit/

# Or skip integration tests explicitly
pytest -m "not integration"
```

### Run Specific Test Files

```bash
# API tests
pytest tests/unit/test_api.py

# State management tests
pytest tests/unit/test_state.py

# Graph node tests
pytest tests/unit/test_graph_nodes.py

# Configuration tests
pytest tests/unit/test_configuration.py
```

### Run with Coverage

```bash
pytest --cov=ollama_deep_researcher --cov-report=html tests/unit/
```

View coverage report:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Run Integration Tests (Requires Ollama)

⚠️ **Prerequisites**:
1. Ollama must be running: `ollama serve`
2. Model must be available: `ollama pull llama3.2`

```bash
# Using just
just intg-test

# Or directly with pytest
pytest tests/intg/

# Or run all tests including integration
pytest
```

### Run Tests with Verbose Output

```bash
pytest -v tests/unit/
```

### Run Specific Test Classes or Functions

```bash
# Run specific test class
pytest tests/unit/test_api.py::TestHealthEndpoint

# Run specific test function
pytest tests/unit/test_api.py::TestHealthEndpoint::test_health_check_returns_200_ok
```

## Test Categories

### Unit Tests
- **No external dependencies** - all services mocked
- Fast execution (< 5 seconds total)
- 80%+ code coverage goal
- Safe to run in CI/CD

### Integration Tests
- **Require Ollama service** to be running
- Slower execution (1-3 minutes per test)
- Test real end-to-end workflows
- Skip in CI with: `pytest -m "not integration"`

## Environment Variables for Testing

Tests use the following environment variables (can be set in `.env` or shell):

```bash
# LLM Configuration
LLM_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434/

# Test-specific overrides
PYTEST_TIMEOUT=300
```

## Writing New Tests

### Example Unit Test

```python
import pytest
from unittest.mock import patch

def test_my_function(mock_graph_success):
    """Test description."""
    with patch("module.function", return_value="mocked"):
        result = my_function()
        assert result == expected
```

### Example Integration Test

```python
@pytest.mark.integration
def test_with_real_ollama():
    """Test requiring Ollama."""
    # Test code here
    pass
```

## Fixtures Available

See `conftest.py` for all available fixtures:

- `mock_graph_success` - Mock successful graph execution
- `mock_graph_failure` - Mock failed graph execution
- `mock_graph_timeout` - Mock timeout scenario
- `mock_graph_exception` - Mock exception scenario
- `mock_ollama_llm` - Mock ChatOllama instance
- `sample_summary_state` - Pre-populated SummaryState
- `mock_config` - Mock RunnableConfig

## Continuous Integration

For CI/CD pipelines, run only unit tests:

```bash
pytest -m "not integration" --cov=ollama_deep_researcher --cov-report=xml
```

## Troubleshooting

### Import Errors

If you see import errors, ensure the package is installed in editable mode:

```bash
pip install -e .
```

### Async Warnings

If you see warnings about async tests, ensure `pytest-asyncio` is installed and `asyncio_mode = "auto"` is set in `pyproject.toml`.

### Integration Tests Fail

1. Check Ollama is running: `curl http://localhost:11434/api/tags`
2. Check model is available: `ollama list`
3. Pull model if needed: `ollama pull llama3.2`

## Test Coverage Goals

- **API Endpoints**: 100%
- **Error Handlers**: 100%
- **Core Business Logic**: 80%+
- **Overall**: 80%+

## Next Steps

After implementation is complete:
1. Run full test suite: `pytest`
2. Check coverage: `pytest --cov=ollama_deep_researcher`
3. Fix any failing tests
4. Add integration tests for new features
5. Set up CI/CD to run tests automatically
