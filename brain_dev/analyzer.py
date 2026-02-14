"""
Analysis logic for Dev Brain.

This module contains the core reasoning logic that operates on
data retrieved from the Context Engine.
"""

import ast
from dataclasses import dataclass, field
from typing import Any, Optional
import hashlib
import re


@dataclass
class CoverageGap:
    """A gap in test coverage."""

    gap_id: str
    pattern: list[str]
    support: float
    priority: str  # low, medium, high, critical
    suggested_test_name: str
    suggested_test_file: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "gap_id": self.gap_id,
            "pattern": self.pattern,
            "support": self.support,
            "priority": self.priority,
            "suggested_test": self.suggested_test_name,
            "suggested_file": self.suggested_test_file,
            "description": self.description,
        }


@dataclass
class MissingBehavior:
    """A user behavior not captured in code/tests."""

    behavior_id: str
    pattern: list[str]
    observed_count: int
    description: str
    suggested_action: str
    affected_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "behavior_id": self.behavior_id,
            "pattern": self.pattern,
            "observed_count": self.observed_count,
            "description": self.description,
            "suggested_action": self.suggested_action,
            "affected_files": self.affected_files,
        }


@dataclass
class SuggestedUnitCase:
    """A suggested test case to write.

    Note: Named SuggestedUnitCase to avoid pytest collection warnings
    (pytest collects classes with "Test" prefix OR suffix).
    """

    test_name: str
    test_file: str
    test_code: str
    covers_pattern: list[str]
    framework: str
    style: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_name": self.test_name,
            "test_file": self.test_file,
            "test_code": self.test_code,
            "covers_pattern": self.covers_pattern,
            "framework": self.framework,
            "style": self.style,
        }


# Backwards compatibility aliases
TestSuggestion = SuggestedUnitCase
GeneratedTest = SuggestedUnitCase


@dataclass
class RefactorSuggestion:
    """A suggested refactoring."""

    suggestion_id: str
    suggestion_type: str  # extract_function, rename, simplify, etc.
    location: str  # file:line
    reason: str
    confidence: float
    code_before: str = ""
    code_after: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "suggestion_id": self.suggestion_id,
            "type": self.suggestion_type,
            "location": self.location,
            "reason": self.reason,
            "confidence": self.confidence,
            "code_before": self.code_before,
            "code_after": self.code_after,
        }


@dataclass
class UXInsight:
    """A UX insight from behavior analysis."""

    insight_id: str
    finding: str
    supporting_patterns: int
    confidence: float
    suggestion: str
    metric: str  # dropoff, time_to_complete, error_rate

    def to_dict(self) -> dict[str, Any]:
        return {
            "insight_id": self.insight_id,
            "finding": self.finding,
            "supporting_patterns": self.supporting_patterns,
            "confidence": self.confidence,
            "suggestion": self.suggestion,
            "metric": self.metric,
        }


class CoverageAnalyzer:
    """Analyzes test coverage gaps."""

    def __init__(self, min_support: float = 0.05):
        self.min_support = min_support

    def analyze_gaps(
        self,
        observed_patterns: list[dict],
        test_patterns: list[list[str]],
    ) -> list[CoverageGap]:
        """
        Compare observed patterns to test coverage.

        Args:
            observed_patterns: Patterns from context_patterns tool
            test_patterns: Patterns covered by existing tests

        Returns:
            List of coverage gaps
        """
        test_pattern_set = {tuple(p) for p in test_patterns}
        gaps = []

        for pattern_data in observed_patterns:
            pattern = tuple(pattern_data.get("sequence", []))
            support = pattern_data.get("support", 0)

            if support < self.min_support:
                continue

            if pattern not in test_pattern_set:
                gap = self._create_gap(pattern, support)
                gaps.append(gap)

        # Sort by priority (support)
        gaps.sort(key=lambda g: g.support, reverse=True)
        return gaps

    def _create_gap(self, pattern: tuple, support: float) -> CoverageGap:
        """Create a CoverageGap from a pattern."""
        gap_id = hashlib.md5(str(pattern).encode()).hexdigest()[:8]

        # Determine priority based on support
        if support >= 0.3:
            priority = "critical"
        elif support >= 0.2:
            priority = "high"
        elif support >= 0.1:
            priority = "medium"
        else:
            priority = "low"

        # Generate suggested test name
        test_name = self._suggest_test_name(pattern)
        test_file = self._suggest_test_file(pattern)

        # Generate description
        description = f"Flow: {' → '.join(pattern)}"

        return CoverageGap(
            gap_id=f"gap_{gap_id}",
            pattern=list(pattern),
            support=support,
            priority=priority,
            suggested_test_name=test_name,
            suggested_test_file=test_file,
            description=description,
        )

    def _suggest_test_name(self, pattern: tuple) -> str:
        """Generate a test name from a pattern."""
        # Extract meaningful parts
        parts = []
        for event_type in pattern:
            # Take the action part (after the dot)
            if "." in event_type:
                parts.append(event_type.split(".")[-1])
            else:
                parts.append(event_type)

        # Combine into test name
        name = "_".join(parts[:3])  # Limit length
        return f"test_{name}_flow"

    def _suggest_test_file(self, pattern: tuple) -> str:
        """Suggest a test file based on pattern."""
        # Look for domain hints in pattern
        first_event = pattern[0] if pattern else "general"

        if "." in first_event:
            domain = first_event.split(".")[0]
        else:
            domain = "general"

        return f"tests/test_{domain}.py"


class BehaviorAnalyzer:
    """Analyzes user behavior patterns."""

    def find_missing_behaviors(
        self,
        observed_patterns: list[dict],
        code_symbols: list[dict],
        min_count: int = 5,
    ) -> list[MissingBehavior]:
        """
        Find behaviors observed but not handled in code.

        Args:
            observed_patterns: Patterns from context_patterns
            code_symbols: Code symbols from context_search_code
            min_count: Minimum occurrences to consider

        Returns:
            List of missing behaviors
        """
        # Extract event types from code
        code_events = self._extract_code_events(code_symbols)

        missing = []
        for pattern_data in observed_patterns:
            pattern = pattern_data.get("sequence", [])
            count = pattern_data.get("occurrence_count", 0)

            if count < min_count:
                continue

            # Check if pattern events are handled in code
            unhandled = [e for e in pattern if e not in code_events]
            if unhandled:
                behavior = self._create_missing_behavior(
                    pattern, count, unhandled
                )
                missing.append(behavior)

        return missing

    def _extract_code_events(self, symbols: list[dict]) -> set[str]:
        """Extract event types that code handles."""
        events = set()

        for symbol in symbols:
            name = symbol.get("name", "").lower()
            # Look for handler patterns
            if "handle" in name or "on_" in name or "process" in name:
                # Extract event type from name
                parts = re.split(r"[_\s]", name)
                for part in parts:
                    if part not in ("handle", "on", "process", "event"):
                        events.add(part)

        return events

    def _create_missing_behavior(
        self, pattern: list, count: int, unhandled: list
    ) -> MissingBehavior:
        """Create a MissingBehavior."""
        behavior_id = hashlib.md5(str(pattern).encode()).hexdigest()[:8]

        return MissingBehavior(
            behavior_id=f"behavior_{behavior_id}",
            pattern=pattern,
            observed_count=count,
            description=f"Pattern observed {count} times with unhandled events: {unhandled}",
            suggested_action=f"Add handlers for: {', '.join(unhandled)}",
        )


class CodeTestGenerator:
    """Generates test suggestions.

    Note: Named CodeTestGenerator (not TestGenerator) to avoid pytest
    collection warnings when the class name starts with "Test".
    """

    TEMPLATES = {
        "pytest": {
            "unit": '''
def {test_name}():
    """Test that {description}."""
    # Arrange: Set up test data

    # Act: Call the function/method

    # Assert: Verify the result
    pass
''',
            "integration": '''
@pytest.mark.integration
def {test_name}():
    """Integration test for {description}."""
    # Note: Add integration test implementation
    pass
''',
        },
        "jest": {
            "unit": '''
describe('{test_name}', () => {{
  it('should {description}', () => {{
    // Arrange: Set up test data

    // Act: Call the function

    // Assert: Verify the result
  }});
}});
''',
        },
    }

    def generate_test(
        self,
        gap: CoverageGap,
        framework: str = "pytest",
        style: str = "unit",
    ) -> TestSuggestion:
        """
        Generate a test suggestion for a coverage gap.

        Args:
            gap: The coverage gap to address
            framework: Test framework (pytest, jest, go)
            style: Test style (unit, integration, e2e)

        Returns:
            TestSuggestion with generated code
        """
        template = self.TEMPLATES.get(framework, {}).get(style, "")

        if not template:
            template = f"# Write {style} test for {gap.pattern}\n    pass"

        test_code = template.format(
            test_name=gap.suggested_test_name,
            description=gap.description,
        )

        return SuggestedUnitCase(
            test_name=gap.suggested_test_name,
            test_file=gap.suggested_test_file,
            test_code=test_code.strip(),
            covers_pattern=gap.pattern,
            framework=framework,
            style=style,
        )


# Backwards compatibility alias (deprecated, use CodeTestGenerator)
TestGenerator = CodeTestGenerator


class RefactorAnalyzer:
    """Analyzes code for refactoring opportunities."""

    def analyze_code(
        self,
        symbols: list[dict],
        patterns: list[dict],
        analysis_type: str = "complexity",
    ) -> list[RefactorSuggestion]:
        """
        Analyze code for refactoring opportunities.

        Args:
            symbols: Code symbols from context
            patterns: Usage patterns
            analysis_type: Type of analysis (complexity, duplication, naming)

        Returns:
            List of refactoring suggestions
        """
        suggestions = []

        if analysis_type == "complexity":
            suggestions.extend(self._analyze_complexity(symbols))
        elif analysis_type == "duplication":
            suggestions.extend(self._analyze_duplication(symbols, patterns))
        elif analysis_type == "naming":
            suggestions.extend(self._analyze_naming(symbols))

        return suggestions

    @staticmethod
    def _ast_complexity(source: str) -> int:
        """Compute AST-based complexity score for a code snippet.

        Complexity formula (inspired by cyclomatic complexity):
          +1 for each: If, For, While, Try, With, ExceptHandler,
                       match/case (MatchCase — Python 3.10+)
          +1 per additional operand in BoolOp (``a and b and c`` = +2)
          +1 for each comprehension node (ListComp, SetComp, DictComp, GeneratorExp)

        Returns 0 when the source cannot be parsed as valid Python.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return 0

        score = 0
        for node in ast.walk(tree):
            if isinstance(node, (
                ast.If, ast.For, ast.While, ast.Try,
                ast.With, ast.ExceptHandler,
            )):
                score += 1
            elif isinstance(node, ast.BoolOp):
                # ``a and b`` has 2 values → +1; ``a and b and c`` → +2
                score += len(node.values) - 1
            elif isinstance(node, (
                ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
            )):
                score += 1
            # Python 3.10+ match/case
            elif hasattr(ast, "MatchCase") and isinstance(node, ast.MatchCase):
                score += 1

        return score

    def _analyze_complexity(self, symbols: list[dict]) -> list[RefactorSuggestion]:
        """Find overly complex functions using AST-based complexity scoring."""
        suggestions = []

        for symbol in symbols:
            code = symbol.get("source_code", "")
            if not code:
                continue

            branches = self._ast_complexity(code)

            if branches > 5:  # Threshold for suggesting refactoring
                suggestions.append(RefactorSuggestion(
                    suggestion_id=f"complex_{symbol.get('name', 'unknown')}",
                    suggestion_type="reduce_complexity",
                    location=f"{symbol.get('file_path', '')}:{symbol.get('line', 0)}",
                    reason=f"Function has {branches} control flow branches (AST-based)",
                    confidence=min(0.5 + (branches - 5) * 0.1, 0.95),
                ))

        return suggestions

    def _analyze_duplication(
        self, symbols: list[dict], patterns: list[dict]
    ) -> list[RefactorSuggestion]:
        """Find duplicated code patterns."""
        # Group by signature similarity
        suggestions = []

        # Simple approach: look for functions with similar names
        name_groups: dict[str, list] = {}
        for symbol in symbols:
            base_name = re.sub(r'\d+$', '', symbol.get("name", ""))
            if base_name not in name_groups:
                name_groups[base_name] = []
            name_groups[base_name].append(symbol)

        for base_name, group in name_groups.items():
            if len(group) > 2:
                suggestions.append(RefactorSuggestion(
                    suggestion_id=f"dup_{base_name}",
                    suggestion_type="extract_common",
                    location=group[0].get("file_path", ""),
                    reason=f"Found {len(group)} similar functions with base name '{base_name}'",
                    confidence=0.6,
                ))

        return suggestions

    def _analyze_naming(self, symbols: list[dict]) -> list[RefactorSuggestion]:
        """Find naming issues."""
        suggestions = []

        for symbol in symbols:
            name = symbol.get("name", "")

            # Check for single-letter names (except in lambdas)
            if len(name) == 1 and symbol.get("symbol_type") != "lambda":
                suggestions.append(RefactorSuggestion(
                    suggestion_id=f"name_{name}_{symbol.get('line', 0)}",
                    suggestion_type="rename",
                    location=f"{symbol.get('file_path', '')}:{symbol.get('line', 0)}",
                    reason=f"Single-letter name '{name}' lacks clarity",
                    confidence=0.7,
                ))

            # Check for very long names
            if len(name) > 50:
                suggestions.append(RefactorSuggestion(
                    suggestion_id=f"name_long_{symbol.get('line', 0)}",
                    suggestion_type="rename",
                    location=f"{symbol.get('file_path', '')}:{symbol.get('line', 0)}",
                    reason=f"Name '{name[:30]}...' is too long ({len(name)} chars)",
                    confidence=0.6,
                ))

        return suggestions


class UXAnalyzer:
    """Analyzes UX patterns from user behavior."""

    def analyze_flow(
        self,
        patterns: list[dict],
        flow_type: str = "general",
        metric: str = "dropoff",
    ) -> list[UXInsight]:
        """
        Analyze user flow patterns for UX insights.

        Args:
            patterns: Behavior patterns from context
            flow_type: Type of flow to analyze
            metric: Metric to focus on

        Returns:
            List of UX insights
        """
        insights = []

        if metric == "dropoff":
            insights.extend(self._analyze_dropoff(patterns, flow_type))
        elif metric == "error_rate":
            insights.extend(self._analyze_errors(patterns, flow_type))

        return insights

    def _analyze_dropoff(
        self, patterns: list[dict], flow_type: str
    ) -> list[UXInsight]:
        """Find high-dropoff points in flows."""
        insights = []

        # Group patterns by prefix
        prefix_counts: dict[tuple, int] = {}
        for pattern in patterns:
            seq = tuple(pattern.get("sequence", []))
            count = pattern.get("occurrence_count", 0)

            for i in range(1, len(seq)):
                prefix = seq[:i]
                if prefix not in prefix_counts:
                    prefix_counts[prefix] = 0
                prefix_counts[prefix] += count

        # Find where counts drop significantly
        for pattern in patterns:
            seq = tuple(pattern.get("sequence", []))
            count = pattern.get("occurrence_count", 0)

            for i in range(1, len(seq)):
                prefix = seq[:i]
                prefix_count = prefix_counts.get(prefix, 0)

                if prefix_count > 0:
                    continuation_rate = count / prefix_count
                    if continuation_rate < 0.5:  # 50% dropoff
                        insight_id = hashlib.md5(str(prefix).encode()).hexdigest()[:8]
                        insights.append(UXInsight(
                            insight_id=f"dropoff_{insight_id}",
                            finding=f"{int((1-continuation_rate)*100)}% of users drop off after step {i}",
                            supporting_patterns=1,
                            confidence=min(0.9, continuation_rate + 0.3),
                            suggestion=f"Investigate friction at step {i}: {prefix[-1] if prefix else 'start'}",
                            metric="dropoff",
                        ))

        return insights

    def _analyze_errors(
        self, patterns: list[dict], flow_type: str
    ) -> list[UXInsight]:
        """Find high-error patterns."""
        insights = []

        for pattern in patterns:
            seq = pattern.get("sequence", [])
            count = pattern.get("occurrence_count", 0)

            # Look for error events
            error_events = [e for e in seq if "error" in e.lower() or "fail" in e.lower()]
            if error_events and count > 5:
                insight_id = hashlib.md5(str(seq).encode()).hexdigest()[:8]
                insights.append(UXInsight(
                    insight_id=f"error_{insight_id}",
                    finding=f"Error flow observed {count} times: {' → '.join(seq)}",
                    supporting_patterns=1,
                    confidence=0.8,
                    suggestion=f"Add error handling or user guidance for: {error_events[0]}",
                    metric="error_rate",
                ))

        return insights


@dataclass
class DocSuggestion:
    """A documentation suggestion."""

    suggestion_id: str
    symbol_name: str
    symbol_type: str  # function, class, module
    location: str
    doc_type: str  # missing, incomplete, outdated
    suggested_doc: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "suggestion_id": self.suggestion_id,
            "symbol_name": self.symbol_name,
            "symbol_type": self.symbol_type,
            "location": self.location,
            "doc_type": self.doc_type,
            "suggested_doc": self.suggested_doc,
            "confidence": self.confidence,
        }


@dataclass
class SecurityIssue:
    """A security issue found in code."""

    issue_id: str
    severity: str  # low, medium, high, critical
    category: str  # injection, xss, auth, crypto, etc.
    location: str
    description: str
    recommendation: str
    confidence: float
    cwe_id: Optional[str] = None  # Common Weakness Enumeration ID

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "severity": self.severity,
            "category": self.category,
            "location": self.location,
            "description": self.description,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "cwe_id": self.cwe_id,
        }


class DocsAnalyzer:
    """Analyzes code for documentation opportunities."""

    def analyze_docs(
        self,
        symbols: list[dict],
        doc_style: str = "google",
    ) -> list[DocSuggestion]:
        """
        Analyze symbols for missing or incomplete documentation.

        Args:
            symbols: Code symbols to analyze
            doc_style: Documentation style (google, numpy, sphinx)

        Returns:
            List of documentation suggestions
        """
        suggestions = []

        for symbol in symbols:
            name = symbol.get("name", "")
            symbol_type = symbol.get("symbol_type", "function")
            docstring = symbol.get("docstring", "")
            file_path = symbol.get("file_path", "")
            line = symbol.get("line", 0)

            # Skip private/dunder methods (except __init__)
            if name.startswith("_") and name != "__init__":
                continue

            suggestion = self._analyze_symbol_docs(
                name, symbol_type, docstring, file_path, line, doc_style
            )
            if suggestion:
                suggestions.append(suggestion)

        return suggestions

    def _analyze_symbol_docs(
        self,
        name: str,
        symbol_type: str,
        docstring: str,
        file_path: str,
        line: int,
        doc_style: str,
    ) -> Optional[DocSuggestion]:
        """Analyze a single symbol's documentation."""
        suggestion_id = hashlib.md5(f"{file_path}:{name}".encode()).hexdigest()[:8]
        location = f"{file_path}:{line}"

        # Missing docstring
        if not docstring:
            return DocSuggestion(
                suggestion_id=f"doc_{suggestion_id}",
                symbol_name=name,
                symbol_type=symbol_type,
                location=location,
                doc_type="missing",
                suggested_doc=self._generate_doc_template(name, symbol_type, doc_style),
                confidence=0.9,
            )

        # Check for incomplete docstring
        issues = self._check_doc_completeness(docstring, symbol_type)
        if issues:
            return DocSuggestion(
                suggestion_id=f"doc_{suggestion_id}",
                symbol_name=name,
                symbol_type=symbol_type,
                location=location,
                doc_type="incomplete",
                suggested_doc=f"Missing: {', '.join(issues)}",
                confidence=0.7,
            )

        return None

    def _generate_doc_template(
        self, name: str, symbol_type: str, doc_style: str
    ) -> str:
        """Generate a documentation template."""
        if doc_style == "google":
            if symbol_type == "function":
                return f'''"""Brief description of {name}.

Args:
    param1: Description of param1.

Returns:
    Description of return value.

Raises:
    ExceptionType: When this exception is raised.
"""'''
            elif symbol_type == "class":
                return f'''"""Brief description of {name}.

Attributes:
    attr1: Description of attr1.
"""'''
        return f'"""Document {name}."""'

    def _check_doc_completeness(self, docstring: str, symbol_type: str) -> list[str]:
        """Check if docstring is complete."""
        issues = []
        doc_lower = docstring.lower()

        if symbol_type == "function":
            # Check for Args section if function likely has parameters
            if "args:" not in doc_lower and "parameters:" not in doc_lower:
                issues.append("Args/Parameters section")
            if "returns:" not in doc_lower and "return" not in doc_lower:
                issues.append("Returns section")

        # Check for very short docstrings
        if len(docstring.strip()) < 20:
            issues.append("detailed description")

        return issues


class SecurityAnalyzer:
    """Analyzes code for security vulnerabilities."""

    # Patterns that may indicate security issues
    SECURITY_PATTERNS = {
        "sql_injection": {
            "patterns": [
                r'execute\s*\(\s*["\'].*["\']\s*%\s',  # %-formatting on string passed to execute
                r'execute\s*\(\s*f["\']',     # f-string in SQL
                r'cursor\.execute\s*\(\s*[^,]+\+',  # String concat in SQL
            ],
            "severity": "critical",
            "cwe": "CWE-89",
            "recommendation": "Use parameterized queries instead of string formatting",
        },
        "command_injection": {
            "patterns": [
                r'os\.system\s*\(',
                r'subprocess\.call\s*\(\s*[^,\[\]]+\+',
                r'subprocess\.run\s*\(\s*shell\s*=\s*True',
                r'eval\s*\(',
                r'exec\s*\(',
            ],
            "severity": "critical",
            "cwe": "CWE-78",
            "recommendation": "Avoid shell=True and validate/sanitize all inputs",
        },
        "hardcoded_secrets": {
            "patterns": [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][A-Za-z0-9]{20,}["\']',
            ],
            "severity": "high",
            "cwe": "CWE-798",
            "recommendation": "Use environment variables or secure vault for secrets",
        },
        "insecure_crypto": {
            "patterns": [
                r'md5\s*\(',
                r'sha1\s*\(',
                r'DES\.',
                r'random\.random\s*\(',  # For crypto purposes
            ],
            "severity": "medium",
            "cwe": "CWE-327",
            "recommendation": "Use modern cryptographic algorithms (SHA-256+, AES)",
        },
        "path_traversal": {
            "patterns": [
                r'open\s*\(\s*[^,]+\+',  # String concat in file path
                r'Path\s*\(\s*[^,]+\+',
            ],
            "severity": "high",
            "cwe": "CWE-22",
            "recommendation": "Validate and sanitize file paths, use Path.resolve()",
        },
        "insecure_deserialization": {
            "patterns": [
                r'pickle\.loads?\s*\(',
                r'yaml\.load\s*\([^,]+\)',  # Without Loader
                r'marshal\.loads?\s*\(',
            ],
            "severity": "critical",
            "cwe": "CWE-502",
            "recommendation": "Avoid deserializing untrusted data, use yaml.safe_load()",
        },
        "xss": {
            "patterns": [
                r'\.innerHTML\s*=',  # Direct innerHTML assignment
                r'document\.write\s*\(',  # document.write with dynamic content
                r'\.html\s*\([^)]*\+',  # jQuery .html() with string concat
                r'dangerouslySetInnerHTML',  # React unsafe HTML
                r'render_template_string\s*\(',  # Flask unsafe template
                r'Markup\s*\([^)]*\+',  # Flask Markup with concat
            ],
            "severity": "high",
            "cwe": "CWE-79",
            "recommendation": "Use proper escaping, avoid innerHTML, use textContent instead",
        },
        "ssrf": {
            "patterns": [
                r'requests\.(get|post|put|delete|patch)\s*\([^)]*\+',  # URL concat
                r'urllib\.request\.urlopen\s*\([^)]*\+',
                r'httpx\.(get|post|put|delete|patch)\s*\([^)]*\+',
                r'aiohttp\.ClientSession\(\)\.get\s*\([^)]*\+',
            ],
            "severity": "high",
            "cwe": "CWE-918",
            "recommendation": "Validate and allowlist URLs, don't allow user-controlled URLs",
        },
        "xxe": {
            "patterns": [
                r'etree\.parse\s*\(',  # XML parsing without safe parser
                r'xml\.dom\.minidom\.parse\s*\(',
                r'xml\.sax\.parse\s*\(',
                r'ElementTree\.parse\s*\(',
            ],
            "severity": "high",
            "cwe": "CWE-611",
            "recommendation": "Disable external entity processing, use defusedxml",
        },
        "log_injection": {
            "patterns": [
                r'logging\.(info|debug|warning|error|critical)\s*\([^)]*\+',
                r'logger\.(info|debug|warning|error|critical)\s*\(f["\']',
            ],
            "severity": "medium",
            "cwe": "CWE-117",
            "recommendation": "Use structured logging, sanitize log inputs",
        },
    }

    # Sink functions where dynamic strings indicate injection risk.
    # Maps (object_attr_or_name) → (category, severity, cwe, recommendation).
    _INJECTION_SINKS: dict[str, tuple[str, str, str, str]] = {
        "execute": (
            "sql_injection", "critical", "CWE-89",
            "Use parameterized queries instead of string formatting",
        ),
        "executemany": (
            "sql_injection", "critical", "CWE-89",
            "Use parameterized queries instead of string formatting",
        ),
        "os.system": (
            "command_injection", "critical", "CWE-78",
            "Avoid os.system(); use subprocess with a list of args",
        ),
        "subprocess.call": (
            "command_injection", "critical", "CWE-78",
            "Avoid shell=True and validate/sanitize all inputs",
        ),
        "subprocess.run": (
            "command_injection", "critical", "CWE-78",
            "Avoid shell=True and validate/sanitize all inputs",
        ),
        "subprocess.Popen": (
            "command_injection", "critical", "CWE-78",
            "Avoid shell=True and validate/sanitize all inputs",
        ),
    }

    @staticmethod
    def _is_dynamic_string(node: ast.expr) -> Optional[str]:
        """Return a human-readable reason if *node* builds a string dynamically.

        Detects:
        * f-strings  (``ast.JoinedStr``)
        * string concatenation  (``ast.BinOp`` with ``Add``)
        * ``.format()`` calls on a string literal
        * ``%`` formatting  (``ast.BinOp`` with ``Mod`` on a string)

        Returns ``None`` when the node is a plain literal or non-string.
        """
        if isinstance(node, ast.JoinedStr):
            return "f-string with interpolated variables"
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Add):
                return "string concatenation (+)"
            if isinstance(node.op, ast.Mod) and isinstance(node.left, ast.Constant):
                return "%-formatting with variables"
        if isinstance(node, ast.Call):
            # "...".format(...)
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "format"
                and isinstance(node.func.value, ast.Constant)
                and isinstance(node.func.value.value, str)
            ):
                return ".format() with variables"
        return None

    @staticmethod
    def _call_target_name(node: ast.Call) -> str:
        """Extract the dotted name of a call target, e.g. ``cursor.execute``."""
        func = node.func
        if isinstance(func, ast.Attribute):
            # one level: obj.method
            if isinstance(func.value, ast.Name):
                return f"{func.value.id}.{func.attr}"
            # deeper chaining — just return the method name
            return func.attr
        if isinstance(func, ast.Name):
            return func.id
        return ""

    def _ast_detect_injections(
        self,
        source: str,
        file_path: str,
        base_line: int,
    ) -> list[SecurityIssue]:
        """Walk the AST looking for dynamic strings passed to known sinks.

        This catches patterns that regex misses:
        * ``query = f"SELECT ... {user}"; cursor.execute(query)``
        * ``cursor.execute("..." + user_input)``
        * ``cursor.execute("...".format(user_input))``
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []  # Fall through to regex fallback

        issues: list[SecurityIssue] = []
        seen_categories: set[str] = set()

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            target = self._call_target_name(node)

            # Check direct sink calls (e.g. cursor.execute, os.system)
            # Match either the full dotted name or just the method name
            sink_info = self._INJECTION_SINKS.get(target)
            if sink_info is None:
                # Try matching just the method part (e.g. "execute")
                method = target.rsplit(".", 1)[-1] if "." in target else target
                sink_info = self._INJECTION_SINKS.get(method)

            if sink_info is None:
                continue

            category, severity, cwe, recommendation = sink_info

            if category in seen_categories:
                continue  # One finding per category

            # Check if any positional argument is a dynamic string
            for arg in node.args:
                reason = self._is_dynamic_string(arg)
                if reason:
                    issue_id = hashlib.md5(
                        f"{file_path}:{base_line}:{category}:ast".encode()
                    ).hexdigest()[:8]
                    issues.append(SecurityIssue(
                        issue_id=f"sec_{issue_id}",
                        severity=severity,
                        category=category,
                        location=f"{file_path}:{base_line}",
                        description=(
                            f"Potential {category.replace('_', ' ')}: "
                            f"{target}() called with {reason}"
                        ),
                        recommendation=recommendation,
                        confidence=0.85,
                        cwe_id=cwe,
                    ))
                    seen_categories.add(category)
                    break

        return issues

    def analyze_security(
        self,
        symbols: list[dict],
        severity_threshold: str = "low",
    ) -> list[SecurityIssue]:
        """
        Analyze code for security vulnerabilities.

        Uses AST-based detection for injection sinks (SQL, command) when the
        source parses successfully, then falls back to regex heuristics for
        patterns not covered by AST (secrets, crypto, XSS, etc.).

        Args:
            symbols: Code symbols with source_code
            severity_threshold: Minimum severity to report

        Returns:
            List of security issues found
        """
        severity_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        threshold = severity_order.get(severity_threshold, 0)

        issues = []

        for symbol in symbols:
            source = symbol.get("source_code", "")
            if not source:
                continue

            file_path = symbol.get("file_path", "")
            line = symbol.get("line", 0)

            # --- Phase 1: AST-based injection detection ---
            ast_issues = self._ast_detect_injections(source, file_path, line)
            ast_categories = {i.category for i in ast_issues}
            for issue in ast_issues:
                if severity_order.get(issue.severity, 0) >= threshold:
                    issues.append(issue)

            # --- Phase 2: Regex fallback for remaining categories ---
            for category, config in self.SECURITY_PATTERNS.items():
                if category in ast_categories:
                    continue  # AST already covered this category

                if severity_order.get(config["severity"], 0) < threshold:
                    continue

                for pattern in config["patterns"]:
                    matches = re.finditer(pattern, source, re.IGNORECASE)
                    for match in matches:
                        issue_id = hashlib.md5(
                            f"{file_path}:{line}:{category}".encode()
                        ).hexdigest()[:8]

                        issues.append(SecurityIssue(
                            issue_id=f"sec_{issue_id}",
                            severity=config["severity"],
                            category=category,
                            location=f"{file_path}:{line}",
                            description=f"Potential {category.replace('_', ' ')} vulnerability detected",
                            recommendation=config["recommendation"],
                            confidence=0.7,
                            cwe_id=config.get("cwe"),
                        ))
                        break  # One issue per category per symbol

        # Sort by severity
        issues.sort(
            key=lambda i: severity_order.get(i.severity, 0),
            reverse=True,
        )

        return issues
