"""Tests for policy guardrails functionality."""

import pytest

from agentic_rag.agent.guardrails import (
    PolicyGuardrail,
    Severity,
    GuardrailResult,
    create_manufacturing_guardrail,
    create_financial_guardrail,
)


class TestPolicyGuardrail:
    """Test suite for PolicyGuardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create a basic guardrail with test rules."""
        g = PolicyGuardrail()
        g.add_rule(
            name="temperature",
            limit=80,
            unit="°C",
            severity=Severity.CRITICAL,
            description="Maximum safe temperature",
            aliases=["temp", "operating_temp"]
        )
        g.add_rule(
            name="pressure",
            limit=100,
            unit="PSI",
            severity=Severity.WARNING,
            description="Maximum pressure",
        )
        return g

    def test_check_within_limit(self, guardrail):
        """Test checking a value within limits."""
        result = guardrail.check("temperature", 75, "°C")

        assert result.is_safe is True
        assert result.severity == Severity.INFO.value
        assert result.metric_name == "temperature"

    def test_check_exceeds_limit(self, guardrail):
        """Test checking a value that exceeds limits."""
        result = guardrail.check("temperature", 85, "°C")

        assert result.is_safe is False
        assert result.severity == Severity.CRITICAL.value
        assert "POLICY VIOLATION" in result.message

    def test_check_exact_limit(self, guardrail):
        """Test checking a value exactly at the limit."""
        result = guardrail.check("temperature", 80, "°C")

        assert result.is_safe is True  # <= is within limit

    def test_check_with_alias(self, guardrail):
        """Test checking using an alias name."""
        result = guardrail.check("temp", 85, "°C")

        assert result.is_safe is False
        assert result.severity == Severity.CRITICAL.value

    def test_check_unknown_metric(self, guardrail):
        """Test checking an unknown metric returns safe."""
        result = guardrail.check("unknown_metric", 1000, "units")

        assert result.is_safe is True
        assert "No policy rule defined" in result.message

    def test_check_string_value(self, guardrail):
        """Test checking a string value with number."""
        result = guardrail.check("temperature", "75°C", "")

        assert result.is_safe is True

    def test_check_string_value_exceeds(self, guardrail):
        """Test checking a string value that exceeds."""
        result = guardrail.check("temperature", "85.5°C", "")

        assert result.is_safe is False

    def test_check_all(self, guardrail):
        """Test checking multiple values at once."""
        values = {
            "temperature": {"value": 75, "unit": "°C"},
            "pressure": {"value": 120, "unit": "PSI"},  # Exceeds limit
        }

        results = guardrail.check_all(values)

        assert len(results) == 2
        temp_result = next(r for r in results if r.metric_name == "temperature")
        pressure_result = next(r for r in results if r.metric_name == "pressure")

        assert temp_result.is_safe is True
        assert pressure_result.is_safe is False

    def test_get_violations(self, guardrail):
        """Test getting only violations."""
        values = {
            "temperature": {"value": 75, "unit": "°C"},
            "pressure": {"value": 120, "unit": "PSI"},
        }

        violations = guardrail.get_violations(values)

        assert len(violations) == 1
        assert violations[0].metric_name == "pressure"


class TestManufacturingGuardrail:
    """Test pre-configured manufacturing guardrail."""

    def test_temperature_check(self):
        """Test temperature limits in manufacturing guardrail."""
        guardrail = create_manufacturing_guardrail()

        safe_result = guardrail.check("temperature", 75, "°C")
        unsafe_result = guardrail.check("temperature", 85, "°C")

        assert safe_result.is_safe is True
        assert unsafe_result.is_safe is False
        assert unsafe_result.severity == Severity.CRITICAL.value

    def test_voltage_check(self):
        """Test voltage limits in manufacturing guardrail."""
        guardrail = create_manufacturing_guardrail()

        safe_result = guardrail.check("voltage", 230, "V")
        unsafe_result = guardrail.check("voltage", 260, "V")

        assert safe_result.is_safe is True
        assert unsafe_result.is_safe is False

    def test_pressure_check(self):
        """Test pressure limits in manufacturing guardrail."""
        guardrail = create_manufacturing_guardrail()

        safe_result = guardrail.check("pressure", 90, "PSI")
        unsafe_result = guardrail.check("pressure", 110, "PSI")

        assert safe_result.is_safe is True
        assert unsafe_result.is_safe is False


class TestFinancialGuardrail:
    """Test pre-configured financial guardrail."""

    def test_budget_check_default(self):
        """Test budget limits with default value."""
        guardrail = create_financial_guardrail()

        safe_result = guardrail.check("budget", 50000, "$")
        unsafe_result = guardrail.check("budget", 150000, "$")

        assert safe_result.is_safe is True
        assert unsafe_result.is_safe is False

    def test_budget_check_custom(self):
        """Test budget limits with custom value."""
        guardrail = create_financial_guardrail(budget_limit=200000)

        result = guardrail.check("budget", 150000, "$")

        assert result.is_safe is True

    def test_variance_check(self):
        """Test variance limits."""
        guardrail = create_financial_guardrail()

        safe_result = guardrail.check("variance", 5, "%")
        unsafe_result = guardrail.check("variance", 15, "%")

        assert safe_result.is_safe is True
        assert unsafe_result.is_safe is False


class TestGuardrailResult:
    """Test GuardrailResult dataclass."""

    def test_result_creation(self):
        """Test creating a GuardrailResult."""
        result = GuardrailResult(
            is_safe=True,
            severity="info",
            message="All good",
            metric_name="test",
            value=42,
            limit=100
        )

        assert result.is_safe is True
        assert result.limit == 100
