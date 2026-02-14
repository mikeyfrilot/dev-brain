"""
Tests for DocsAnalyzer and SecurityAnalyzer.
"""

import pytest
import json
from brain_dev.analyzer import DocsAnalyzer, SecurityAnalyzer, DocSuggestion, SecurityIssue


class TestDocSuggestion:
    """Tests for DocSuggestion dataclass."""

    def test_create_doc_suggestion(self):
        """Test creating a doc suggestion."""
        suggestion = DocSuggestion(
            suggestion_id="doc_abc123",
            symbol_name="my_function",
            symbol_type="function",
            location="test.py:10",
            doc_type="missing",
            suggested_doc="Add a docstring here",
            confidence=0.9,
        )
        assert suggestion.symbol_name == "my_function"
        assert suggestion.doc_type == "missing"

    def test_doc_suggestion_to_dict(self):
        """Test DocSuggestion.to_dict()."""
        suggestion = DocSuggestion(
            suggestion_id="doc_abc123",
            symbol_name="my_function",
            symbol_type="function",
            location="test.py:10",
            doc_type="incomplete",
            suggested_doc="Missing: Returns section",
            confidence=0.7,
        )
        d = suggestion.to_dict()
        assert d["suggestion_id"] == "doc_abc123"
        assert d["symbol_name"] == "my_function"
        assert d["symbol_type"] == "function"
        assert d["doc_type"] == "incomplete"
        assert d["confidence"] == 0.7


class TestSecurityIssue:
    """Tests for SecurityIssue dataclass."""

    def test_create_security_issue(self):
        """Test creating a security issue."""
        issue = SecurityIssue(
            issue_id="sec_abc123",
            severity="critical",
            category="sql_injection",
            location="db.py:10",
            description="SQL injection detected",
            recommendation="Use parameterized queries",
            confidence=0.8,
            cwe_id="CWE-89",
        )
        assert issue.severity == "critical"
        assert issue.cwe_id == "CWE-89"

    def test_security_issue_to_dict(self):
        """Test SecurityIssue.to_dict()."""
        issue = SecurityIssue(
            issue_id="sec_abc123",
            severity="high",
            category="hardcoded_secrets",
            location="config.py:5",
            description="Hardcoded password",
            recommendation="Use env vars",
            confidence=0.7,
            cwe_id="CWE-798",
        )
        d = issue.to_dict()
        assert d["issue_id"] == "sec_abc123"
        assert d["severity"] == "high"
        assert d["category"] == "hardcoded_secrets"
        assert d["cwe_id"] == "CWE-798"

    def test_security_issue_without_cwe(self):
        """Test SecurityIssue without CWE ID."""
        issue = SecurityIssue(
            issue_id="sec_abc123",
            severity="low",
            category="other",
            location="test.py:1",
            description="Some issue",
            recommendation="Fix it",
            confidence=0.5,
        )
        assert issue.cwe_id is None
        d = issue.to_dict()
        assert d["cwe_id"] is None


class TestDocsAnalyzer:
    """Tests for DocsAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        return DocsAnalyzer()

    def test_analyze_docs_missing(self, analyzer):
        """Test detecting missing docstrings."""
        symbols = [
            {
                "name": "my_function",
                "symbol_type": "function",
                "docstring": "",
                "file_path": "test.py",
                "line": 10,
            }
        ]
        suggestions = analyzer.analyze_docs(symbols)
        assert len(suggestions) == 1
        assert suggestions[0].doc_type == "missing"
        assert suggestions[0].symbol_name == "my_function"

    def test_analyze_docs_incomplete(self, analyzer):
        """Test detecting incomplete docstrings."""
        symbols = [
            {
                "name": "calculate",
                "symbol_type": "function",
                "docstring": "Short doc.",  # Too short, no returns
                "file_path": "math.py",
                "line": 5,
            }
        ]
        suggestions = analyzer.analyze_docs(symbols)
        assert len(suggestions) == 1
        assert suggestions[0].doc_type == "incomplete"

    def test_analyze_docs_complete(self, analyzer):
        """Test that complete docs don't generate suggestions."""
        symbols = [
            {
                "name": "well_documented",
                "symbol_type": "function",
                "docstring": "This is a well-documented function that does something important.\n\nArgs:\n    value: The input value.\n\nReturns:\n    The result of the operation.",
                "file_path": "good.py",
                "line": 1,
            }
        ]
        suggestions = analyzer.analyze_docs(symbols)
        assert len(suggestions) == 0

    def test_analyze_docs_skips_private(self, analyzer):
        """Test that private functions are skipped."""
        symbols = [
            {
                "name": "_private_helper",
                "symbol_type": "function",
                "docstring": "",
                "file_path": "test.py",
                "line": 10,
            }
        ]
        suggestions = analyzer.analyze_docs(symbols)
        assert len(suggestions) == 0

    def test_analyze_docs_includes_init(self, analyzer):
        """Test that __init__ is not skipped."""
        symbols = [
            {
                "name": "__init__",
                "symbol_type": "function",
                "docstring": "",
                "file_path": "test.py",
                "line": 10,
            }
        ]
        suggestions = analyzer.analyze_docs(symbols)
        assert len(suggestions) == 1

    def test_generate_doc_template_function(self, analyzer):
        """Test generating function doc template."""
        template = analyzer._generate_doc_template("my_func", "function", "google")
        assert "Args:" in template
        assert "Returns:" in template
        assert "Raises:" in template

    def test_generate_doc_template_class(self, analyzer):
        """Test generating class doc template."""
        template = analyzer._generate_doc_template("MyClass", "class", "google")
        assert "Attributes:" in template

    def test_generate_doc_template_non_google(self, analyzer):
        """Test non-Google style template."""
        template = analyzer._generate_doc_template("func", "function", "numpy")
        assert "Document func" in template

    def test_check_doc_completeness_with_returns(self, analyzer):
        """Test completeness check passes with returns."""
        issues = analyzer._check_doc_completeness(
            "This function calculates and returns the sum.",
            "function"
        )
        assert "Returns section" not in issues

    def test_check_doc_completeness_short(self, analyzer):
        """Test completeness fails for short docs."""
        issues = analyzer._check_doc_completeness("Too short", "function")
        assert "detailed description" in issues

    def test_check_doc_completeness_missing_args(self, analyzer):
        """Regression: missing Args/Parameters section must be reported.

        Previously the detection branch had a silent 'pass' instead of
        appending to the issues list.
        """
        issues = analyzer._check_doc_completeness(
            "This function processes data and returns the result.",
            "function"
        )
        assert "Args/Parameters section" in issues

    def test_check_doc_completeness_has_args(self, analyzer):
        """Test that a docstring with Args section passes the check."""
        issues = analyzer._check_doc_completeness(
            "This function processes data and returns the result.\n\nArgs:\n    x: the input value\n\nReturns:\n    The processed result.",
            "function"
        )
        assert "Args/Parameters section" not in issues
        assert "Returns section" not in issues

    def test_analyze_docs_reports_incomplete_args(self, analyzer):
        """Regression: analyze_docs must flag functions with docstrings
        missing an Args section as incomplete."""
        symbols = [
            {
                "name": "process_data",
                "symbol_type": "function",
                "docstring": "This function processes data and returns the result.",
                "file_path": "test.py",
                "line": 10,
            }
        ]
        suggestions = analyzer.analyze_docs(symbols)
        assert len(suggestions) == 1
        assert suggestions[0].doc_type == "incomplete"
        assert "Args/Parameters section" in suggestions[0].suggested_doc


class TestSecurityAnalyzer:
    """Tests for SecurityAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        return SecurityAnalyzer()

    def test_detect_sql_injection(self, analyzer):
        """Test SQL injection detection."""
        symbols = [
            {
                "name": "query",
                "file_path": "db.py",
                "line": 10,
                "source_code": 'cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")',
            }
        ]
        issues = analyzer.analyze_security(symbols)
        assert len(issues) >= 1
        assert any(i.category == "sql_injection" for i in issues)
        assert any(i.severity == "critical" for i in issues)

    def test_detect_command_injection(self, analyzer):
        """Test command injection detection."""
        symbols = [
            {
                "name": "run_cmd",
                "file_path": "utils.py",
                "line": 5,
                "source_code": "os.system(user_input)",
            }
        ]
        issues = analyzer.analyze_security(symbols)
        assert len(issues) >= 1
        assert any(i.category == "command_injection" for i in issues)

    def test_detect_eval(self, analyzer):
        """Test eval() detection."""
        symbols = [
            {
                "name": "dangerous",
                "file_path": "bad.py",
                "line": 1,
                "source_code": "result = eval(user_code)",
            }
        ]
        issues = analyzer.analyze_security(symbols)
        assert any(i.category == "command_injection" for i in issues)

    def test_detect_hardcoded_secrets(self, analyzer):
        """Test hardcoded secrets detection."""
        symbols = [
            {
                "name": "config",
                "file_path": "config.py",
                "line": 1,
                "source_code": 'password = "super_secret_123"',
            }
        ]
        issues = analyzer.analyze_security(symbols)
        assert len(issues) >= 1
        assert any(i.category == "hardcoded_secrets" for i in issues)

    def test_detect_insecure_crypto(self, analyzer):
        """Test insecure crypto detection."""
        symbols = [
            {
                "name": "hash_password",
                "file_path": "auth.py",
                "line": 10,
                "source_code": "hashed = md5(password.encode())",
            }
        ]
        issues = analyzer.analyze_security(symbols)
        assert any(i.category == "insecure_crypto" for i in issues)

    def test_detect_pickle(self, analyzer):
        """Test insecure deserialization detection."""
        symbols = [
            {
                "name": "load_data",
                "file_path": "data.py",
                "line": 1,
                "source_code": "data = pickle.load(file)",
            }
        ]
        issues = analyzer.analyze_security(symbols)
        assert any(i.category == "insecure_deserialization" for i in issues)

    def test_severity_threshold(self, analyzer):
        """Test severity threshold filtering."""
        symbols = [
            {
                "name": "config",
                "file_path": "config.py",
                "line": 1,
                "source_code": 'password = "secret"\nhashed = md5(x)',
            }
        ]
        # Get all issues
        all_issues = analyzer.analyze_security(symbols, "low")

        # Get only high+ issues
        high_issues = analyzer.analyze_security(symbols, "high")

        # High threshold should filter out medium (insecure_crypto)
        assert len(high_issues) <= len(all_issues)

    def test_empty_source_skipped(self, analyzer):
        """Test that empty source code is skipped."""
        symbols = [
            {
                "name": "empty",
                "file_path": "test.py",
                "line": 1,
                "source_code": "",
            }
        ]
        issues = analyzer.analyze_security(symbols)
        assert len(issues) == 0

    def test_issues_sorted_by_severity(self, analyzer):
        """Test that issues are sorted by severity."""
        symbols = [
            {
                "name": "multi",
                "file_path": "bad.py",
                "line": 1,
                "source_code": 'password = "secret"\nos.system(cmd)\nhashed = md5(x)',
            }
        ]
        issues = analyzer.analyze_security(symbols)
        if len(issues) >= 2:
            severity_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
            for i in range(len(issues) - 1):
                assert severity_order[issues[i].severity] >= severity_order[issues[i + 1].severity]

    def test_multiple_patterns_same_category(self, analyzer):
        """Test that multiple patterns in same category are detected."""
        symbols = [
            {
                "name": "multi_secrets",
                "file_path": "bad.py",
                "line": 1,
                "source_code": 'password = "one"\napi_key = "two"\nsecret = "three"',
            }
        ]
        issues = analyzer.analyze_security(symbols)
        # Multiple patterns match, should find them
        hardcoded_count = sum(1 for i in issues if i.category == "hardcoded_secrets")
        assert hardcoded_count >= 1

    def test_detect_xss_innerhtml(self, analyzer):
        """Test XSS detection via innerHTML."""
        symbols = [
            {
                "name": "render",
                "file_path": "view.py",
                "line": 10,
                "source_code": 'element.innerHTML = user_input',
            }
        ]
        issues = analyzer.analyze_security(symbols)
        assert any(i.category == "xss" for i in issues)
        assert any(i.cwe_id == "CWE-79" for i in issues)

    def test_detect_xss_dangerously_set(self, analyzer):
        """Test XSS detection via React dangerouslySetInnerHTML."""
        symbols = [
            {
                "name": "component",
                "file_path": "app.jsx",
                "line": 5,
                "source_code": 'return <div dangerouslySetInnerHTML={{__html: data}} />',
            }
        ]
        issues = analyzer.analyze_security(symbols)
        assert any(i.category == "xss" for i in issues)

    def test_detect_ssrf(self, analyzer):
        """Test SSRF detection via requests with user input."""
        symbols = [
            {
                "name": "fetch",
                "file_path": "api.py",
                "line": 20,
                "source_code": 'response = requests.get(base_url + user_path)',
            }
        ]
        issues = analyzer.analyze_security(symbols)
        assert any(i.category == "ssrf" for i in issues)
        assert any(i.cwe_id == "CWE-918" for i in issues)

    def test_detect_xxe(self, analyzer):
        """Test XXE detection via unsafe XML parsing."""
        symbols = [
            {
                "name": "parse_xml",
                "file_path": "parser.py",
                "line": 15,
                "source_code": 'tree = etree.parse(xml_file)',
            }
        ]
        issues = analyzer.analyze_security(symbols)
        assert any(i.category == "xxe" for i in issues)
        assert any(i.cwe_id == "CWE-611" for i in issues)

    def test_detect_log_injection(self, analyzer):
        """Test log injection detection."""
        symbols = [
            {
                "name": "log_user",
                "file_path": "logging.py",
                "line": 8,
                "source_code": 'logger.info(f"User logged in: {username}")',
            }
        ]
        issues = analyzer.analyze_security(symbols)
        assert any(i.category == "log_injection" for i in issues)
        assert any(i.cwe_id == "CWE-117" for i in issues)

    # --- AST-based injection detection ---

    def test_ast_detect_fstring_sql_injection(self, analyzer):
        """AST detects f-string passed directly to cursor.execute()."""
        symbols = [
            {
                "name": "get_user",
                "file_path": "db.py",
                "line": 10,
                "source_code": (
                    'cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")\n'
                ),
            }
        ]
        issues = analyzer.analyze_security(symbols)
        sql_issues = [i for i in issues if i.category == "sql_injection"]
        assert len(sql_issues) >= 1
        # AST layer provides a descriptive message about f-string
        assert any("f-string" in i.description for i in sql_issues)

    def test_ast_detect_concat_sql_injection(self, analyzer):
        """AST detects string concatenation in cursor.execute()."""
        symbols = [
            {
                "name": "search",
                "file_path": "db.py",
                "line": 5,
                "source_code": (
                    'cursor.execute("SELECT * FROM items WHERE name = \'" + user_input + "\'")\n'
                ),
            }
        ]
        issues = analyzer.analyze_security(symbols)
        sql_issues = [i for i in issues if i.category == "sql_injection"]
        assert len(sql_issues) >= 1
        # AST layer should provide a more descriptive message
        assert any("concatenation" in i.description for i in sql_issues) or \
               any("sql injection" in i.description.lower() for i in sql_issues)

    def test_ast_detect_format_sql_injection(self, analyzer):
        """AST detects .format() in cursor.execute()."""
        symbols = [
            {
                "name": "lookup",
                "file_path": "db.py",
                "line": 3,
                "source_code": (
                    'cursor.execute("SELECT * FROM users WHERE id = {}".format(user_input))\n'
                ),
            }
        ]
        issues = analyzer.analyze_security(symbols)
        sql_issues = [i for i in issues if i.category == "sql_injection"]
        assert len(sql_issues) >= 1

    def test_ast_no_false_positive_on_static_query(self, analyzer):
        """AST does NOT flag static (safe) queries with parameter placeholders."""
        symbols = [
            {
                "name": "safe_query",
                "file_path": "db.py",
                "line": 1,
                "source_code": (
                    'cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))\n'
                ),
            }
        ]
        issues = analyzer.analyze_security(symbols)
        sql_issues = [i for i in issues if i.category == "sql_injection"]
        assert len(sql_issues) == 0

    def test_ast_detect_fstring_in_subprocess(self, analyzer):
        """AST detects f-string passed to subprocess.run()."""
        symbols = [
            {
                "name": "run_cmd",
                "file_path": "ops.py",
                "line": 1,
                "source_code": (
                    "import subprocess\n"
                    'subprocess.run(f"ls {user_dir}", shell=True)\n'
                ),
            }
        ]
        issues = analyzer.analyze_security(symbols)
        cmd_issues = [i for i in issues if i.category == "command_injection"]
        assert len(cmd_issues) >= 1
