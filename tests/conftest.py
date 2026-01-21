"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
    This is a sample document for testing the RAG pipeline.

    It contains multiple paragraphs with different content.

    The third paragraph discusses temperature limits.
    Maximum operating temperature is 80°C.

    The fourth paragraph has financial data.
    Budget variance was +$3,600 or approximately 1%.
    """


@pytest.fixture
def sample_table_json():
    """Sample table JSON for testing."""
    return {
        "headers": ["Parameter", "Value", "Limit", "Status"],
        "rows": [
            ["Temperature", "72°C", "80°C", "WARNING"],
            ["Voltage", "235V", "240V", "OK"],
            ["Current", "28A", "32A", "OK"],
        ]
    }


@pytest.fixture
def data_dir():
    """Path to test data directory."""
    return Path(__file__).parent.parent / "data"
