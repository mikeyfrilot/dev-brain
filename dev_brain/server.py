"""
MCP Server implementation for Dev Brain.

Provides 5 tools for developer insights:
- coverage_analyze: Analyze test coverage gaps
- behavior_missing: Find unhandled user behaviors
- tests_generate: Generate test suggestions
- refactor_suggest: Suggest refactoring opportunities
- ux_insights: Extract UX insights from behavior
"""

import asyncio
import json
from typing import Any, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    Resource,
)

from .config import DevBrainConfig
from .analyzer import (
    CoverageAnalyzer,
    BehaviorAnalyzer,
    TestGenerator,
    RefactorAnalyzer,
    UXAnalyzer,
    DocsAnalyzer,
    SecurityAnalyzer,
    CoverageGap,
)
from .smart_test_generator import generate_tests_for_file


def create_server(config: Optional[DevBrainConfig] = None) -> Server:
    """
    Create and configure the Dev Brain MCP server.

    Args:
        config: Configuration (uses defaults if None)

    Returns:
        Configured MCP Server
    """
    config = config or DevBrainConfig()
    server = Server(config.server_name)

    # Initialize analyzers
    _coverage_analyzer: Optional[CoverageAnalyzer] = None
    _behavior_analyzer: Optional[BehaviorAnalyzer] = None
    _test_generator: Optional[TestGenerator] = None
    _refactor_analyzer: Optional[RefactorAnalyzer] = None
    _ux_analyzer: Optional[UXAnalyzer] = None

    def get_coverage_analyzer() -> CoverageAnalyzer:
        nonlocal _coverage_analyzer
        if _coverage_analyzer is None:
            _coverage_analyzer = CoverageAnalyzer(min_support=config.min_gap_support)
        return _coverage_analyzer

    def get_behavior_analyzer() -> BehaviorAnalyzer:
        nonlocal _behavior_analyzer
        if _behavior_analyzer is None:
            _behavior_analyzer = BehaviorAnalyzer()
        return _behavior_analyzer

    def get_test_generator() -> TestGenerator:
        nonlocal _test_generator
        if _test_generator is None:
            _test_generator = TestGenerator()
        return _test_generator

    def get_refactor_analyzer() -> RefactorAnalyzer:
        nonlocal _refactor_analyzer
        if _refactor_analyzer is None:
            _refactor_analyzer = RefactorAnalyzer()
        return _refactor_analyzer

    def get_ux_analyzer() -> UXAnalyzer:
        nonlocal _ux_analyzer
        if _ux_analyzer is None:
            _ux_analyzer = UXAnalyzer()
        return _ux_analyzer

    _docs_analyzer: Optional[DocsAnalyzer] = None
    _security_analyzer: Optional[SecurityAnalyzer] = None

    def get_docs_analyzer() -> DocsAnalyzer:
        nonlocal _docs_analyzer
        if _docs_analyzer is None:
            _docs_analyzer = DocsAnalyzer()
        return _docs_analyzer

    def get_security_analyzer() -> SecurityAnalyzer:
        nonlocal _security_analyzer
        if _security_analyzer is None:
            _security_analyzer = SecurityAnalyzer()
        return _security_analyzer

    # =========================================================================
    # Tool Definitions
    # =========================================================================

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available tools."""
        return [
            Tool(
                name="coverage_analyze",
                description="Analyze test coverage gaps by comparing observed user flows to test coverage",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "patterns": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sequence": {"type": "array", "items": {"type": "string"}},
                                    "support": {"type": "number"},
                                    "occurrence_count": {"type": "integer"},
                                },
                            },
                            "description": "Observed patterns (from context_patterns)"
                        },
                        "test_patterns": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "description": "Patterns covered by existing tests"
                        },
                        "min_support": {
                            "type": "number",
                            "default": 0.05,
                            "description": "Minimum support threshold for gaps"
                        },
                    },
                    "required": ["patterns"],
                },
            ),
            Tool(
                name="behavior_missing",
                description="Find user behaviors not captured in code or tests",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "patterns": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "Observed behavior patterns"
                        },
                        "code_symbols": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "Code symbols (from context_search_code)"
                        },
                        "min_count": {
                            "type": "integer",
                            "default": 5,
                            "description": "Minimum occurrence count to consider"
                        },
                    },
                    "required": ["patterns"],
                },
            ),
            Tool(
                name="tests_generate",
                description="Generate test suggestions for coverage gaps",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "gap": {
                            "type": "object",
                            "properties": {
                                "pattern": {"type": "array", "items": {"type": "string"}},
                                "support": {"type": "number"},
                                "description": {"type": "string"},
                            },
                            "description": "Coverage gap to generate test for"
                        },
                        "framework": {
                            "type": "string",
                            "enum": ["pytest", "jest", "go"],
                            "default": "pytest",
                            "description": "Test framework"
                        },
                        "style": {
                            "type": "string",
                            "enum": ["unit", "integration", "e2e"],
                            "default": "unit",
                            "description": "Test style"
                        },
                    },
                    "required": ["gap"],
                },
            ),
            Tool(
                name="refactor_suggest",
                description="Suggest refactoring based on code and usage patterns",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbols": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "Code symbols to analyze"
                        },
                        "patterns": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "Usage patterns (optional)"
                        },
                        "analysis_type": {
                            "type": "string",
                            "enum": ["complexity", "duplication", "naming", "all"],
                            "default": "all",
                            "description": "Type of analysis"
                        },
                    },
                    "required": ["symbols"],
                },
            ),
            Tool(
                name="ux_insights",
                description="Extract UX insights from user behavior patterns",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "patterns": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "Behavior patterns to analyze"
                        },
                        "flow_type": {
                            "type": "string",
                            "default": "general",
                            "description": "Type of flow (search, checkout, onboarding)"
                        },
                        "metric": {
                            "type": "string",
                            "enum": ["dropoff", "error_rate", "all"],
                            "default": "all",
                            "description": "Metric to analyze"
                        },
                    },
                    "required": ["patterns"],
                },
            ),
            Tool(
                name="brain_stats",
                description="Get Dev Brain server statistics",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="smart_tests_generate",
                description="Generate complete pytest test file for a Python source file using AST analysis. Creates tests with proper mocks, fixtures, and assertions.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute path to the Python source file to generate tests for"
                        },
                    },
                    "required": ["file_path"],
                },
            ),
            Tool(
                name="docs_generate",
                description="Generate documentation suggestions for code symbols with missing or incomplete docstrings",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbols": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "Code symbols to analyze for documentation"
                        },
                        "doc_style": {
                            "type": "string",
                            "enum": ["google", "numpy", "sphinx"],
                            "default": "google",
                            "description": "Documentation style to use"
                        },
                    },
                    "required": ["symbols"],
                },
            ),
            Tool(
                name="security_audit",
                description="Analyze code for security vulnerabilities (SQL injection, command injection, hardcoded secrets, etc.)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbols": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "Code symbols with source_code to analyze"
                        },
                        "severity_threshold": {
                            "type": "string",
                            "enum": ["low", "medium", "high", "critical"],
                            "default": "low",
                            "description": "Minimum severity level to report"
                        },
                    },
                    "required": ["symbols"],
                },
            ),
        ]

    # =========================================================================
    # Tool Implementations
    # =========================================================================

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle tool calls."""

        if name == "coverage_analyze":
            return await handle_coverage_analyze(arguments)

        elif name == "behavior_missing":
            return await handle_behavior_missing(arguments)

        elif name == "tests_generate":
            return await handle_tests_generate(arguments)

        elif name == "refactor_suggest":
            return await handle_refactor_suggest(arguments)

        elif name == "ux_insights":
            return await handle_ux_insights(arguments)

        elif name == "brain_stats":
            return await handle_brain_stats(arguments)

        elif name == "smart_tests_generate":
            return await handle_smart_tests_generate(arguments)

        elif name == "docs_generate":
            return await handle_docs_generate(arguments)

        elif name == "security_audit":
            return await handle_security_audit(arguments)

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    async def handle_coverage_analyze(args: dict) -> list[TextContent]:
        """Handle coverage_analyze tool."""
        analyzer = get_coverage_analyzer()

        patterns = args.get("patterns", [])
        test_patterns = args.get("test_patterns", [])
        min_support = args.get("min_support", config.min_gap_support)

        # Update analyzer threshold
        analyzer.min_support = min_support

        # Analyze gaps
        gaps = analyzer.analyze_gaps(patterns, test_patterns)

        # Calculate coverage
        total_patterns = len(patterns)
        covered = total_patterns - len(gaps)
        coverage_pct = (covered / total_patterns * 100) if total_patterns > 0 else 0

        return [TextContent(
            type="text",
            text=json.dumps({
                "total_flows": total_patterns,
                "covered_flows": covered,
                "coverage_percentage": round(coverage_pct, 1),
                "gaps_found": len(gaps),
                "gaps": [g.to_dict() for g in gaps[:config.max_suggestions]],
            })
        )]

    async def handle_behavior_missing(args: dict) -> list[TextContent]:
        """Handle behavior_missing tool."""
        analyzer = get_behavior_analyzer()

        patterns = args.get("patterns", [])
        code_symbols = args.get("code_symbols", [])
        min_count = args.get("min_count", 5)

        missing = analyzer.find_missing_behaviors(patterns, code_symbols, min_count)

        return [TextContent(
            type="text",
            text=json.dumps({
                "missing_behaviors": [m.to_dict() for m in missing[:config.max_suggestions]],
                "total_found": len(missing),
            })
        )]

    async def handle_tests_generate(args: dict) -> list[TextContent]:
        """Handle tests_generate tool."""
        generator = get_test_generator()

        gap_data = args.get("gap", {})
        framework = args.get("framework", config.default_test_framework)
        style = args.get("style", config.test_style)

        # Convert gap data to CoverageGap
        from .analyzer import CoverageGap
        import hashlib

        pattern = gap_data.get("pattern", [])
        support = gap_data.get("support", 0.1)

        gap = CoverageGap(
            gap_id=gap_data.get("gap_id", hashlib.md5(str(pattern).encode()).hexdigest()[:8]),
            pattern=pattern,
            support=support,
            priority="medium",
            suggested_test_name=gap_data.get("suggested_test", f"test_{'_'.join(pattern[:2])}_flow"),
            suggested_test_file=gap_data.get("suggested_file", "tests/test_generated.py"),
            description=gap_data.get("description", f"Test for {' → '.join(pattern)}"),
        )

        suggestion = generator.generate_test(gap, framework, style)

        return [TextContent(
            type="text",
            text=json.dumps(suggestion.to_dict())
        )]

    async def handle_refactor_suggest(args: dict) -> list[TextContent]:
        """Handle refactor_suggest tool."""
        analyzer = get_refactor_analyzer()

        symbols = args.get("symbols", [])
        patterns = args.get("patterns", [])
        analysis_type = args.get("analysis_type", "all")

        suggestions = []

        if analysis_type == "all":
            for atype in ["complexity", "duplication", "naming"]:
                suggestions.extend(analyzer.analyze_code(symbols, patterns, atype))
        else:
            suggestions = analyzer.analyze_code(symbols, patterns, analysis_type)

        # Sort by confidence
        suggestions.sort(key=lambda s: s.confidence, reverse=True)

        return [TextContent(
            type="text",
            text=json.dumps({
                "suggestions": [s.to_dict() for s in suggestions[:config.max_suggestions]],
                "total_found": len(suggestions),
            })
        )]

    async def handle_ux_insights(args: dict) -> list[TextContent]:
        """Handle ux_insights tool."""
        analyzer = get_ux_analyzer()

        patterns = args.get("patterns", [])
        flow_type = args.get("flow_type", "general")
        metric = args.get("metric", "all")

        insights = []

        if metric == "all":
            for m in ["dropoff", "error_rate"]:
                insights.extend(analyzer.analyze_flow(patterns, flow_type, m))
        else:
            insights = analyzer.analyze_flow(patterns, flow_type, metric)

        # Sort by confidence
        insights.sort(key=lambda i: i.confidence, reverse=True)

        return [TextContent(
            type="text",
            text=json.dumps({
                "insights": [i.to_dict() for i in insights[:config.max_suggestions]],
                "total_found": len(insights),
                "flow_type": flow_type,
                "metric": metric,
            })
        )]

    async def handle_brain_stats(args: dict) -> list[TextContent]:
        """Handle brain_stats tool."""
        return [TextContent(
            type="text",
            text=json.dumps({
                "server_name": config.server_name,
                "server_version": config.server_version,
                "min_gap_support": config.min_gap_support,
                "min_confidence": config.min_confidence,
                "max_suggestions": config.max_suggestions,
                "default_test_framework": config.default_test_framework,
                "tools_available": 9,
            })
        )]

    async def handle_smart_tests_generate(args: dict) -> list[TextContent]:
        """Handle smart_tests_generate tool."""
        from pathlib import Path

        file_path = args.get("file_path", "")

        # Validate file_path is provided and is a string
        if not file_path or not isinstance(file_path, str):
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": "file_path is required and must be a string",
                    "success": False,
                })
            )]

        # Normalize and resolve the path to prevent path traversal display issues
        try:
            resolved_path = Path(file_path).resolve()
            file_name = resolved_path.name  # Safe: only the filename for error messages
        except (ValueError, OSError):
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": "Invalid file path provided",
                    "success": False,
                })
            )]

        # Validate it's a Python file
        if not file_name.endswith(".py"):
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": "File must be a Python (.py) file",
                    "success": False,
                })
            )]

        # Validate file exists
        if not resolved_path.is_file():
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": f"File not found: {file_name}",
                    "success": False,
                })
            )]

        try:
            # Generate tests using the smart test generator
            test_code = generate_tests_for_file(str(resolved_path))

            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "file_path": str(resolved_path),
                    "file_name": file_name,
                    "test_code": test_code,
                    "lines": len(test_code.split("\n")),
                })
            )]
        except SyntaxError as e:
            # Handle Python syntax errors in the source file
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": f"Syntax error in source file: {e.msg}",
                    "success": False,
                    "file_path": str(resolved_path),
                    "file_name": file_name,
                })
            )]
        except Exception as e:
            # Sanitize error message - don't expose internal paths or stack traces
            error_type = type(e).__name__
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": f"Failed to generate tests: {error_type}",
                    "success": False,
                    "file_path": str(resolved_path),
                    "file_name": file_name,
                })
            )]

    async def handle_docs_generate(args: dict) -> list[TextContent]:
        """Handle docs_generate tool."""
        analyzer = get_docs_analyzer()

        symbols = args.get("symbols", [])
        doc_style = args.get("doc_style", "google")

        suggestions = analyzer.analyze_docs(symbols, doc_style)

        return [TextContent(
            type="text",
            text=json.dumps({
                "suggestions": [s.to_dict() for s in suggestions[:config.max_suggestions]],
                "total_found": len(suggestions),
                "doc_style": doc_style,
            })
        )]

    async def handle_security_audit(args: dict) -> list[TextContent]:
        """Handle security_audit tool."""
        analyzer = get_security_analyzer()

        symbols = args.get("symbols", [])
        severity_threshold = args.get("severity_threshold", "low")

        issues = analyzer.analyze_security(symbols, severity_threshold)

        # Group by severity for summary
        severity_counts = {}
        for issue in issues:
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1

        return [TextContent(
            type="text",
            text=json.dumps({
                "issues": [i.to_dict() for i in issues[:config.max_suggestions]],
                "total_found": len(issues),
                "severity_counts": severity_counts,
                "severity_threshold": severity_threshold,
            })
        )]

    # =========================================================================
    # Resources
    # =========================================================================

    @server.list_resources()
    async def list_resources() -> list[Resource]:
        """List available resources."""
        return [
            Resource(
                uri="brain://stats",
                name="Dev Brain Stats",
                description="Server statistics and configuration",
                mimeType="application/json",
            ),
        ]

    @server.read_resource()
    async def read_resource(uri: str) -> str:
        """Read a resource."""
        if uri == "brain://stats":
            return json.dumps({
                "server_name": config.server_name,
                "server_version": config.server_version,
                "analyzers": ["coverage", "behavior", "test_generator", "refactor", "ux"],
            })

        return json.dumps({"error": "Unknown resource"})

    return server


async def run_server():
    """Run the MCP server."""
    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    """Main entry point."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
