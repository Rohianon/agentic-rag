"""Policy guardrails for safety and compliance checking."""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


class Severity(Enum):
    """Risk severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""

    is_safe: bool
    severity: str
    message: str
    metric_name: str
    value: Any
    limit: Any = None


@dataclass
class PolicyRule:
    """A single policy rule definition."""

    name: str
    description: str
    check_fn: Callable[[Any], bool]
    limit: Any
    unit: str
    severity: Severity


class PolicyGuardrail:
    """
    Policy-based guardrail system for validating extracted values.

    Use cases:
    1. Temperature limits (manufacturing specs)
    2. Financial thresholds (budget limits)
    3. Compliance checks (regulatory limits)
    4. Safety boundaries (operational limits)

    Example:
        guardrail = PolicyGuardrail()
        guardrail.add_rule(
            name="max_temperature",
            limit=80,
            unit="°C",
            severity=Severity.CRITICAL,
            description="Maximum operating temperature"
        )

        result = guardrail.check("temperature", 85, "°C")
        # result.is_safe = False, result.severity = "critical"
    """

    def __init__(self):
        self.rules: dict[str, PolicyRule] = {}
        self._aliases: dict[str, str] = {}

    def add_rule(
        self,
        name: str,
        limit: float | int,
        unit: str = "",
        severity: Severity = Severity.WARNING,
        description: str = "",
        comparison: str = "max",  # "max", "min", "range"
        aliases: list[str] | None = None,
    ):
        """
        Add a policy rule.

        Args:
            name: Rule identifier (e.g., "temperature", "voltage")
            limit: The threshold value
            unit: Unit of measurement
            severity: How critical a violation is
            description: Human-readable description
            comparison: "max" (value must be <= limit), "min" (>=), "range" (tuple)
            aliases: Alternative names that map to this rule
        """
        if comparison == "max":
            check_fn = lambda v: self._parse_number(v) <= limit
        elif comparison == "min":
            check_fn = lambda v: self._parse_number(v) >= limit
        elif comparison == "range":
            # limit should be (min, max) tuple
            check_fn = lambda v: limit[0] <= self._parse_number(v) <= limit[1]
        else:
            check_fn = lambda v: True

        rule = PolicyRule(
            name=name,
            description=description,
            check_fn=check_fn,
            limit=limit,
            unit=unit,
            severity=severity,
        )

        self.rules[name.lower()] = rule

        # Register aliases
        if aliases:
            for alias in aliases:
                self._aliases[alias.lower()] = name.lower()

    def _parse_number(self, value: Any) -> float:
        """Parse a value to a number, handling various formats."""
        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            # Remove common units and symbols
            cleaned = re.sub(r'[°%$,]', '', value)
            # Extract number
            match = re.search(r'-?\d+\.?\d*', cleaned)
            if match:
                return float(match.group())

        raise ValueError(f"Cannot parse '{value}' as number")

    def check(
        self,
        metric_name: str,
        value: Any,
        unit: str = "",
    ) -> GuardrailResult:
        """
        Check a value against policy rules.

        Args:
            metric_name: The name of the metric (e.g., "temperature")
            value: The value to check
            unit: The unit of the value

        Returns:
            GuardrailResult with safety status and details
        """
        # Normalize metric name
        lookup_name = metric_name.lower()

        # Check aliases
        if lookup_name in self._aliases:
            lookup_name = self._aliases[lookup_name]

        # Find matching rule
        rule = self.rules.get(lookup_name)

        if not rule:
            # No rule for this metric - assume safe
            return GuardrailResult(
                is_safe=True,
                severity=Severity.INFO.value,
                message=f"No policy rule defined for '{metric_name}'",
                metric_name=metric_name,
                value=value,
            )

        try:
            is_safe = rule.check_fn(value)
        except (ValueError, TypeError) as e:
            return GuardrailResult(
                is_safe=True,  # Can't check, so don't flag
                severity=Severity.INFO.value,
                message=f"Could not validate '{metric_name}': {e}",
                metric_name=metric_name,
                value=value,
            )

        if is_safe:
            return GuardrailResult(
                is_safe=True,
                severity=Severity.INFO.value,
                message=f"{metric_name} ({value} {unit}) is within limits",
                metric_name=metric_name,
                value=value,
                limit=rule.limit,
            )
        else:
            return GuardrailResult(
                is_safe=False,
                severity=rule.severity.value,
                message=f"POLICY VIOLATION: {metric_name} ({value} {unit}) exceeds limit of {rule.limit} {rule.unit}. {rule.description}",
                metric_name=metric_name,
                value=value,
                limit=rule.limit,
            )

    def check_all(self, values: dict[str, Any]) -> list[GuardrailResult]:
        """Check multiple values at once."""
        results = []
        for metric_name, value_info in values.items():
            if isinstance(value_info, dict):
                value = value_info.get("value")
                unit = value_info.get("unit", "")
            else:
                value = value_info
                unit = ""

            results.append(self.check(metric_name, value, unit))

        return results

    def get_violations(self, values: dict[str, Any]) -> list[GuardrailResult]:
        """Return only the violations from checking values."""
        results = self.check_all(values)
        return [r for r in results if not r.is_safe]


# Pre-configured guardrails for common domains
def create_manufacturing_guardrail() -> PolicyGuardrail:
    """Create guardrails for manufacturing domain."""
    guardrail = PolicyGuardrail()

    guardrail.add_rule(
        name="temperature",
        limit=80,
        unit="°C",
        severity=Severity.CRITICAL,
        description="Maximum operating temperature for standard components",
        aliases=["temp", "operating_temperature", "max_temp"],
    )

    guardrail.add_rule(
        name="voltage",
        limit=250,
        unit="V",
        severity=Severity.CRITICAL,
        description="Maximum voltage rating",
        aliases=["operating_voltage", "max_voltage"],
    )

    guardrail.add_rule(
        name="pressure",
        limit=100,
        unit="PSI",
        severity=Severity.WARNING,
        description="Maximum operating pressure",
        aliases=["operating_pressure", "max_pressure"],
    )

    return guardrail


def create_financial_guardrail(budget_limit: float = 100000) -> PolicyGuardrail:
    """Create guardrails for financial domain."""
    guardrail = PolicyGuardrail()

    guardrail.add_rule(
        name="budget",
        limit=budget_limit,
        unit="$",
        severity=Severity.WARNING,
        description=f"Budget limit of ${budget_limit:,}",
        aliases=["cost", "total_cost", "expense"],
    )

    guardrail.add_rule(
        name="variance",
        limit=10,
        unit="%",
        severity=Severity.WARNING,
        description="Maximum acceptable variance from estimate",
        aliases=["budget_variance", "cost_variance"],
    )

    return guardrail
