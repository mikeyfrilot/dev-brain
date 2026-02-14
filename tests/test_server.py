"""
Tests for Dev Brain MCP Server.
"""

import json
import pytest
from unittest.mock import MagicMock, AsyncMock

from mcp.types import (
    Tool,
    TextContent,
    Resource,
    ListToolsRequest,
    CallToolRequest,
    ListResourcesRequest,
    ReadResourceRequest,
)

from brain_dev.config import DevBrainConfig
from brain_dev.server import create_server, TOOL_DEFINITIONS


# =============================================================================
# Helper Functions
# =============================================================================

async def list_tools(server) -> list[Tool]:
    """List tools from server."""
    handler = server.request_handlers[ListToolsRequest]
    request = MagicMock()
    result = await handler(request)
    return result.root.tools


async def call_tool(server, name: str, arguments: dict) -> list[TextContent]:
    """Call a tool on the server."""
    handler = server.request_handlers[CallToolRequest]
    request = MagicMock()
    request.params = MagicMock()
    request.params.name = name
    request.params.arguments = arguments
    result = await handler(request)
    return result.root.content


async def list_resources(server) -> list[Resource]:
    """List resources from server."""
    handler = server.request_handlers[ListResourcesRequest]
    request = MagicMock()
    result = await handler(request)
    return result.root.resources


async def read_resource(server, uri: str) -> str:
    """Read a resource from server."""
    handler = server.request_handlers[ReadResourceRequest]
    request = MagicMock()
    request.params = MagicMock()
    request.params.uri = uri
    result = await handler(request)
    # MCP wraps the result, extract the text content
    contents = result.root.contents
    if contents and len(contents) > 0:
        return contents[0].text
    return ""


# =============================================================================
# Server Creation Tests
# =============================================================================

class TestServerCreation:
    """Tests for server creation."""

    def test_create_server_default_config(self):
        """Test creating server with default config."""
        server = create_server()
        assert server is not None
        assert server.name == "brain-dev"

    def test_create_server_custom_config(self, config):
        """Test creating server with custom config."""
        server = create_server(config)
        assert server is not None

    def test_server_has_request_handlers(self, server):
        """Test that server has request handlers registered."""
        assert ListToolsRequest in server.request_handlers
        assert CallToolRequest in server.request_handlers
        assert ListResourcesRequest in server.request_handlers
        assert ReadResourceRequest in server.request_handlers


# =============================================================================
# Tool Listing Tests
# =============================================================================

class TestListTools:
    """Tests for listing tools."""

    @pytest.mark.asyncio
    async def test_list_tools_returns_all_tools(self, server):
        """Test that all expected tools are listed."""
        tools = await list_tools(server)

        tool_names = {t.name for t in tools}
        expected = {
            "coverage_analyze",
            "behavior_missing",
            "tests_generate",
            "refactor_suggest",
            "ux_insights",
            "brain_stats",
            "smart_tests_generate",
            "docs_generate",
            "security_audit",
        }
        assert tool_names == expected

    @pytest.mark.asyncio
    async def test_tool_registry_matches_listed_tools(self, server):
        """Regression: TOOL_DEFINITIONS registry must match listed tools exactly."""
        tools = await list_tools(server)

        registry_names = {t.name for t in TOOL_DEFINITIONS}
        listed_names = {t.name for t in tools}
        assert registry_names == listed_names
        assert len(TOOL_DEFINITIONS) == len(tools)

    @pytest.mark.asyncio
    async def test_brain_stats_reports_correct_tool_count(self, server):
        """Regression: brain_stats must reflect actual tool count, not hardcoded."""
        result = await call_tool(server, "brain_stats", {})
        data = json.loads(result[0].text)
        assert data["tools_available"] == len(TOOL_DEFINITIONS)

    @pytest.mark.asyncio
    async def test_tools_have_descriptions(self, server):
        """Test that all tools have descriptions."""
        tools = await list_tools(server)

        for tool in tools:
            assert tool.description is not None
            assert len(tool.description) > 0

    @pytest.mark.asyncio
    async def test_tools_have_input_schemas(self, server):
        """Test that all tools have input schemas."""
        tools = await list_tools(server)

        for tool in tools:
            assert tool.inputSchema is not None
            assert "type" in tool.inputSchema
            assert tool.inputSchema["type"] == "object"


# =============================================================================
# coverage_analyze Tool Tests
# =============================================================================

class TestCoverageAnalyzeTool:
    """Tests for coverage_analyze tool."""

    @pytest.mark.asyncio
    async def test_coverage_analyze_basic(self, server, sample_patterns):
        """Test basic coverage analysis."""
        result = await call_tool(server, "coverage_analyze", {
            "patterns": sample_patterns,
        })

        assert len(result) == 1
        data = json.loads(result[0].text)

        assert "total_flows" in data
        assert "covered_flows" in data
        assert "coverage_percentage" in data
        assert "gaps_found" in data
        assert "gaps" in data

    @pytest.mark.asyncio
    async def test_coverage_analyze_with_test_patterns(
        self, server, sample_patterns, sample_test_patterns
    ):
        """Test coverage analysis with test patterns."""
        result = await call_tool(server, "coverage_analyze", {
            "patterns": sample_patterns,
            "test_patterns": sample_test_patterns,
        })

        data = json.loads(result[0].text)

        # With some tests, coverage should be > 0
        assert data["covered_flows"] > 0
        assert data["coverage_percentage"] > 0

    @pytest.mark.asyncio
    async def test_coverage_analyze_custom_min_support(
        self, server, sample_patterns
    ):
        """Test coverage analysis with custom min_support."""
        result = await call_tool(server, "coverage_analyze", {
            "patterns": sample_patterns,
            "min_support": 0.20,
        })

        data = json.loads(result[0].text)

        # Higher threshold should find fewer gaps
        for gap in data["gaps"]:
            assert gap["support"] >= 0.20

    @pytest.mark.asyncio
    async def test_coverage_analyze_empty_patterns(self, server):
        """Test coverage analysis with empty patterns."""
        result = await call_tool(server, "coverage_analyze", {
            "patterns": [],
        })

        data = json.loads(result[0].text)
        assert data["total_flows"] == 0
        assert data["gaps_found"] == 0


# =============================================================================
# behavior_missing Tool Tests
# =============================================================================

class TestBehaviorMissingTool:
    """Tests for behavior_missing tool."""

    @pytest.mark.asyncio
    async def test_behavior_missing_basic(self, server, sample_patterns):
        """Test basic missing behavior detection."""
        result = await call_tool(server, "behavior_missing", {
            "patterns": sample_patterns,
        })

        assert len(result) == 1
        data = json.loads(result[0].text)

        assert "missing_behaviors" in data
        assert "total_found" in data

    @pytest.mark.asyncio
    async def test_behavior_missing_with_code_symbols(
        self, server, sample_patterns, sample_code_symbols
    ):
        """Test missing behavior with code symbols."""
        result = await call_tool(server, "behavior_missing", {
            "patterns": sample_patterns,
            "code_symbols": sample_code_symbols,
        })

        data = json.loads(result[0].text)
        assert "missing_behaviors" in data

    @pytest.mark.asyncio
    async def test_behavior_missing_custom_min_count(
        self, server, sample_patterns
    ):
        """Test missing behavior with custom min_count."""
        result = await call_tool(server, "behavior_missing", {
            "patterns": sample_patterns,
            "min_count": 50,
        })

        data = json.loads(result[0].text)

        # Higher threshold should find fewer missing behaviors
        for behavior in data["missing_behaviors"]:
            assert behavior["observed_count"] >= 50

    @pytest.mark.asyncio
    async def test_behavior_missing_empty_patterns(self, server):
        """Test missing behavior with empty patterns."""
        result = await call_tool(server, "behavior_missing", {
            "patterns": [],
        })

        data = json.loads(result[0].text)
        assert data["missing_behaviors"] == []


# =============================================================================
# tests_generate Tool Tests
# =============================================================================

class TestTestsGenerateTool:
    """Tests for tests_generate tool."""

    @pytest.mark.asyncio
    async def test_tests_generate_basic(self, server, sample_gap):
        """Test basic test generation."""
        result = await call_tool(server, "tests_generate", {
            "gap": sample_gap,
        })

        assert len(result) == 1
        data = json.loads(result[0].text)

        assert "test_name" in data
        assert "test_file" in data
        assert "test_code" in data
        assert "framework" in data

    @pytest.mark.asyncio
    async def test_tests_generate_pytest(self, server, sample_gap):
        """Test generating pytest tests."""
        result = await call_tool(server, "tests_generate", {
            "gap": sample_gap,
            "framework": "pytest",
        })

        data = json.loads(result[0].text)
        assert data["framework"] == "pytest"
        assert "def test_" in data["test_code"]

    @pytest.mark.asyncio
    async def test_tests_generate_jest(self, server, sample_gap):
        """Test generating Jest tests."""
        result = await call_tool(server, "tests_generate", {
            "gap": sample_gap,
            "framework": "jest",
        })

        data = json.loads(result[0].text)
        assert data["framework"] == "jest"

    @pytest.mark.asyncio
    async def test_tests_generate_go(self, server, sample_gap):
        """Test generating Go tests."""
        result = await call_tool(server, "tests_generate", {
            "gap": sample_gap,
            "framework": "go",
        })

        data = json.loads(result[0].text)
        assert data["framework"] == "go"
        # Go template not defined, falls back to TODO comment
        assert len(data["test_code"]) > 0

    @pytest.mark.asyncio
    async def test_tests_generate_integration_style(self, server, sample_gap):
        """Test generating integration style tests."""
        result = await call_tool(server, "tests_generate", {
            "gap": sample_gap,
            "style": "integration",
        })

        data = json.loads(result[0].text)
        assert data["style"] == "integration"

    @pytest.mark.asyncio
    async def test_tests_generate_e2e_style(self, server, sample_gap):
        """Test generating e2e style tests."""
        result = await call_tool(server, "tests_generate", {
            "gap": sample_gap,
            "style": "e2e",
        })

        data = json.loads(result[0].text)
        assert data["style"] == "e2e"


# =============================================================================
# refactor_suggest Tool Tests
# =============================================================================

class TestRefactorSuggestTool:
    """Tests for refactor_suggest tool."""

    @pytest.mark.asyncio
    async def test_refactor_suggest_basic(self, server, sample_code_symbols):
        """Test basic refactoring suggestions."""
        result = await call_tool(server, "refactor_suggest", {
            "symbols": sample_code_symbols,
        })

        assert len(result) == 1
        data = json.loads(result[0].text)

        assert "suggestions" in data
        assert "total_found" in data

    @pytest.mark.asyncio
    async def test_refactor_suggest_complexity(self, server, sample_code_symbols):
        """Test complexity analysis."""
        result = await call_tool(server, "refactor_suggest", {
            "symbols": sample_code_symbols,
            "analysis_type": "complexity",
        })

        data = json.loads(result[0].text)

        for s in data["suggestions"]:
            assert s["suggestion_type"] == "complexity"

    @pytest.mark.asyncio
    async def test_refactor_suggest_duplication(self, server, sample_code_symbols):
        """Test duplication analysis."""
        result = await call_tool(server, "refactor_suggest", {
            "symbols": sample_code_symbols,
            "analysis_type": "duplication",
        })

        data = json.loads(result[0].text)

        for s in data["suggestions"]:
            assert s["suggestion_type"] == "duplication"

    @pytest.mark.asyncio
    async def test_refactor_suggest_naming(self, server, sample_code_symbols):
        """Test naming analysis."""
        result = await call_tool(server, "refactor_suggest", {
            "symbols": sample_code_symbols,
            "analysis_type": "naming",
        })

        data = json.loads(result[0].text)

        for s in data["suggestions"]:
            assert s["suggestion_type"] == "naming"

    @pytest.mark.asyncio
    async def test_refactor_suggest_with_patterns(
        self, server, sample_code_symbols, sample_patterns
    ):
        """Test refactoring with usage patterns."""
        result = await call_tool(server, "refactor_suggest", {
            "symbols": sample_code_symbols,
            "patterns": sample_patterns,
        })

        data = json.loads(result[0].text)
        assert "suggestions" in data

    @pytest.mark.asyncio
    async def test_refactor_suggest_empty_symbols(self, server):
        """Test refactoring with empty symbols."""
        result = await call_tool(server, "refactor_suggest", {
            "symbols": [],
        })

        data = json.loads(result[0].text)
        assert data["suggestions"] == []


# =============================================================================
# ux_insights Tool Tests
# =============================================================================

class TestUXInsightsTool:
    """Tests for ux_insights tool."""

    @pytest.mark.asyncio
    async def test_ux_insights_basic(self, server, sample_patterns):
        """Test basic UX insights."""
        result = await call_tool(server, "ux_insights", {
            "patterns": sample_patterns,
        })

        assert len(result) == 1
        data = json.loads(result[0].text)

        assert "insights" in data
        assert "total_found" in data
        assert "flow_type" in data
        assert "metric" in data

    @pytest.mark.asyncio
    async def test_ux_insights_dropoff(self, server, sample_patterns):
        """Test dropoff metric analysis."""
        result = await call_tool(server, "ux_insights", {
            "patterns": sample_patterns,
            "metric": "dropoff",
        })

        data = json.loads(result[0].text)
        assert data["metric"] == "dropoff"

        for i in data["insights"]:
            assert i["metric"] == "dropoff"

    @pytest.mark.asyncio
    async def test_ux_insights_error_rate(self, server, sample_patterns):
        """Test error rate metric analysis."""
        result = await call_tool(server, "ux_insights", {
            "patterns": sample_patterns,
            "metric": "error_rate",
        })

        data = json.loads(result[0].text)
        assert data["metric"] == "error_rate"

        for i in data["insights"]:
            assert i["metric"] == "error_rate"

    @pytest.mark.asyncio
    async def test_ux_insights_checkout_flow(self, server, sample_patterns):
        """Test checkout flow analysis."""
        result = await call_tool(server, "ux_insights", {
            "patterns": sample_patterns,
            "flow_type": "checkout",
        })

        data = json.loads(result[0].text)
        assert data["flow_type"] == "checkout"

    @pytest.mark.asyncio
    async def test_ux_insights_search_flow(self, server, sample_patterns):
        """Test search flow analysis."""
        result = await call_tool(server, "ux_insights", {
            "patterns": sample_patterns,
            "flow_type": "search",
        })

        data = json.loads(result[0].text)
        assert data["flow_type"] == "search"

    @pytest.mark.asyncio
    async def test_ux_insights_empty_patterns(self, server):
        """Test UX insights with empty patterns."""
        result = await call_tool(server, "ux_insights", {
            "patterns": [],
        })

        data = json.loads(result[0].text)
        assert data["insights"] == []


# =============================================================================
# brain_stats Tool Tests
# =============================================================================

class TestBrainStatsTool:
    """Tests for brain_stats tool."""

    @pytest.mark.asyncio
    async def test_brain_stats(self, server):
        """Test getting server stats."""
        result = await call_tool(server, "brain_stats", {})

        assert len(result) == 1
        data = json.loads(result[0].text)

        assert "server_name" in data
        assert "server_version" in data
        assert "tools_available" in data
        assert data["tools_available"] == 9

    @pytest.mark.asyncio
    async def test_brain_stats_config_values(self, server):
        """Test that stats include config values."""
        result = await call_tool(server, "brain_stats", {})

        data = json.loads(result[0].text)

        assert "min_gap_support" in data
        assert "min_confidence" in data
        assert "max_suggestions" in data
        assert "default_test_framework" in data


# =============================================================================
# Unknown Tool Tests
# =============================================================================

class TestUnknownTool:
    """Tests for unknown tool handling."""

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self, server):
        """Test that unknown tool returns error message."""
        result = await call_tool(server, "unknown_tool", {})

        assert len(result) == 1
        assert "Unknown tool" in result[0].text


# =============================================================================
# Resource Tests
# =============================================================================

class TestResources:
    """Tests for resources."""

    @pytest.mark.asyncio
    async def test_list_resources(self, server):
        """Test listing resources."""
        resources = await list_resources(server)

        assert len(resources) == 1

        uris = {str(r.uri) for r in resources}
        assert "brain://stats" in uris

    @pytest.mark.asyncio
    async def test_read_brain_stats_resource(self, server):
        """Test reading brain stats resource."""
        text = await read_resource(server, "brain://stats")

        data = json.loads(text)
        assert "server_name" in data
        assert "server_version" in data
        assert "analyzers" in data
        assert len(data["analyzers"]) == 5

    @pytest.mark.asyncio
    async def test_read_unknown_resource(self, server):
        """Test reading unknown resource."""
        text = await read_resource(server, "brain://unknown")

        data = json.loads(text)
        assert "error" in data


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for typical workflows."""

    @pytest.mark.asyncio
    async def test_coverage_to_test_generation_flow(
        self, server, sample_patterns, sample_test_patterns
    ):
        """Test typical flow: analyze coverage -> generate tests."""
        # Step 1: Analyze coverage
        coverage_result = await call_tool(server, "coverage_analyze", {
            "patterns": sample_patterns,
            "test_patterns": sample_test_patterns,
        })

        coverage_data = json.loads(coverage_result[0].text)

        # Step 2: Generate tests for each gap
        if coverage_data["gaps"]:
            gap = coverage_data["gaps"][0]

            test_result = await call_tool(server, "tests_generate", {
                "gap": gap,
                "framework": "pytest",
                "style": "integration",
            })

            test_data = json.loads(test_result[0].text)

            assert test_data["test_name"] is not None
            assert test_data["test_code"] is not None
            assert "def test_" in test_data["test_code"]

    @pytest.mark.asyncio
    async def test_behavior_analysis_flow(
        self, server, sample_patterns, sample_code_symbols
    ):
        """Test behavior analysis flow."""
        # Step 1: Find missing behaviors
        missing_result = await call_tool(server, "behavior_missing", {
            "patterns": sample_patterns,
            "code_symbols": sample_code_symbols,
        })

        missing_data = json.loads(missing_result[0].text)

        # Step 2: Get refactoring suggestions
        refactor_result = await call_tool(server, "refactor_suggest", {
            "symbols": sample_code_symbols,
            "patterns": sample_patterns,
        })

        refactor_data = json.loads(refactor_result[0].text)

        # Both should return valid data
        assert "missing_behaviors" in missing_data
        assert "suggestions" in refactor_data

    @pytest.mark.asyncio
    async def test_ux_analysis_flow(self, server, sample_patterns):
        """Test UX analysis flow."""
        # Analyze multiple metrics
        dropoff_result = await call_tool(server, "ux_insights", {
            "patterns": sample_patterns,
            "flow_type": "checkout",
            "metric": "dropoff",
        })

        error_result = await call_tool(server, "ux_insights", {
            "patterns": sample_patterns,
            "flow_type": "checkout",
            "metric": "error_rate",
        })

        dropoff_data = json.loads(dropoff_result[0].text)
        error_data = json.loads(error_result[0].text)

        assert dropoff_data["flow_type"] == "checkout"
        assert error_data["flow_type"] == "checkout"
        assert dropoff_data["metric"] == "dropoff"
        assert error_data["metric"] == "error_rate"

    @pytest.mark.asyncio
    async def test_full_analysis_workflow(
        self, server, sample_patterns, sample_test_patterns, sample_code_symbols
    ):
        """Test complete analysis workflow."""
        # Step 1: Get coverage gaps
        coverage = await call_tool(server, "coverage_analyze", {
            "patterns": sample_patterns,
            "test_patterns": sample_test_patterns,
        })

        # Step 2: Find missing behaviors
        behaviors = await call_tool(server, "behavior_missing", {
            "patterns": sample_patterns,
            "code_symbols": sample_code_symbols,
        })

        # Step 3: Get refactoring suggestions
        refactoring = await call_tool(server, "refactor_suggest", {
            "symbols": sample_code_symbols,
            "patterns": sample_patterns,
        })

        # Step 4: Get UX insights
        ux = await call_tool(server, "ux_insights", {
            "patterns": sample_patterns,
            "flow_type": "general",
        })

        # Step 5: Get stats
        stats = await call_tool(server, "brain_stats", {})

        # Verify all results are valid
        coverage_data = json.loads(coverage[0].text)
        behaviors_data = json.loads(behaviors[0].text)
        refactoring_data = json.loads(refactoring[0].text)
        ux_data = json.loads(ux[0].text)
        stats_data = json.loads(stats[0].text)

        assert "gaps" in coverage_data
        assert "missing_behaviors" in behaviors_data
        assert "suggestions" in refactoring_data
        assert "insights" in ux_data
        assert "tools_available" in stats_data


# =============================================================================
# Input Validation Tests
# =============================================================================

def _parse_tool_result(result) -> dict:
    """Parse a tool result, handling both JSON and plain-text MCP validation errors."""
    text = result[0].text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # MCP framework returned a plain-text validation error
        return {"error": text, "success": False}


class TestInputValidation:
    """Tests for input validation on all tool handlers.

    MCP validates against inputSchema before handlers run, so missing/wrong-type
    args get caught at the framework level.  Our handlers also have defense-in-depth
    validation.  Either layer may fire; both must produce a clean error (no crash).
    """

    @pytest.fixture
    def server(self):
        return create_server()

    # -- Missing required arguments --

    @pytest.mark.asyncio
    async def test_coverage_analyze_missing_patterns(self, server):
        result = await call_tool(server, "coverage_analyze", {})
        data = _parse_tool_result(result)
        assert data["success"] is False
        assert "patterns" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_behavior_missing_missing_patterns(self, server):
        result = await call_tool(server, "behavior_missing", {})
        data = _parse_tool_result(result)
        assert data["success"] is False
        assert "patterns" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_tests_generate_missing_gap(self, server):
        result = await call_tool(server, "tests_generate", {})
        data = _parse_tool_result(result)
        assert data["success"] is False
        assert "gap" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_refactor_suggest_missing_symbols(self, server):
        result = await call_tool(server, "refactor_suggest", {})
        data = _parse_tool_result(result)
        assert data["success"] is False
        assert "symbols" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_ux_insights_missing_patterns(self, server):
        result = await call_tool(server, "ux_insights", {})
        data = _parse_tool_result(result)
        assert data["success"] is False
        assert "patterns" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_smart_tests_generate_missing_file_path(self, server):
        result = await call_tool(server, "smart_tests_generate", {})
        data = _parse_tool_result(result)
        assert data["success"] is False
        assert "file_path" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_docs_generate_missing_symbols(self, server):
        result = await call_tool(server, "docs_generate", {})
        data = _parse_tool_result(result)
        assert data["success"] is False
        assert "symbols" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_security_audit_missing_symbols(self, server):
        result = await call_tool(server, "security_audit", {})
        data = _parse_tool_result(result)
        assert data["success"] is False
        assert "symbols" in data["error"].lower()

    # -- Wrong type arguments --

    @pytest.mark.asyncio
    async def test_coverage_analyze_wrong_type(self, server):
        result = await call_tool(server, "coverage_analyze", {"patterns": "not-a-list"})
        data = _parse_tool_result(result)
        assert data["success"] is False
        assert "array" in data["error"].lower() or "list" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_tests_generate_wrong_type(self, server):
        result = await call_tool(server, "tests_generate", {"gap": "not-a-dict"})
        data = _parse_tool_result(result)
        assert data["success"] is False
        assert "object" in data["error"].lower() or "dict" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_smart_tests_generate_wrong_type(self, server):
        result = await call_tool(server, "smart_tests_generate", {"file_path": 12345})
        data = _parse_tool_result(result)
        assert data["success"] is False
        assert "string" in data["error"].lower() or "type" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_refactor_suggest_wrong_type(self, server):
        result = await call_tool(server, "refactor_suggest", {"symbols": "not-a-list"})
        data = _parse_tool_result(result)
        assert data["success"] is False
        assert "array" in data["error"].lower() or "list" in data["error"].lower()

    # -- brain_stats accepts empty args (no required fields) --

    @pytest.mark.asyncio
    async def test_brain_stats_no_args_ok(self, server):
        result = await call_tool(server, "brain_stats", {})
        data = json.loads(result[0].text)
        assert "tools_available" in data
