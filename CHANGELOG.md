# Changelog

All notable changes to Dev Brain will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.2] - 2026-02-14

### Changed
- **`confidence` field renamed to `signal_strength`** on RefactorSuggestion,
  UXInsight, DocSuggestion, and SecurityIssue.  The old `confidence` key is
  still emitted in `to_dict()` output and accessible via a `@property` alias
  for backward compatibility.
- `DevBrainConfig.min_confidence` renamed to `min_signal_strength`
  (property alias preserved).
- Complexity scoring now uses real AST-based analysis instead of
  string counting.
- SecurityAnalyzer runs AST-first injection detection before regex
  fallback.
- Regex patterns are precompiled at class-definition time.
- Stdlib whitelist expanded to 100+ modules with `importlib.util.find_spec`
  fallback.

### Fixed
- CI coverage workflow failure caused by orphaned submodule entries.
- Windows temp-path handling in smart test generator (uses `tempfile.gettempdir()`).
- False positive on parameterized SQL queries in security scanner.
- False positive on installed packages detected as stdlib.

### Added
- CODEOWNERS, issue templates, PR template.
- Dependabot config (pip + github-actions, weekly).
- `workflow_dispatch` on CI workflows for manual runs.
- Path filters on CI to skip docs-only changes.
- `uv.lock` committed; CI uses `uv sync --frozen` for deterministic installs.
- Upper bounds on all direct dependencies (`mcp<2`, `pytest<10`, etc.).
- Lock-drift CI check (`uv lock --check`) on pyproject.toml / uv.lock changes.
- `__version__` now reads from `importlib.metadata` (single source of truth).
- Version test asserting `brain_dev.__version__` matches `pyproject.toml`.
- Compatibility policy documented in README (semver, Python support window).
- Build & smoke-test CI workflow: builds wheel, installs in clean env, imports.

## [1.0.0] - 2026-01-24

### Added
- **9 MCP Tools** for code intelligence
  - `coverage_analyze` - Test coverage gap detection
  - `behavior_missing` - Find unhandled edge cases
  - `refactor_suggest` - Complexity and duplication analysis
  - `ux_insights` - UX pattern extraction
  - `tests_generate` - Test case suggestions
  - `smart_tests_generate` - AST-based pytest generation
  - `docs_generate` - Documentation templates
  - `security_audit` - OWASP vulnerability scanning
  - `brain_stats` - Server statistics and health
- **Security Detection** across 9 vulnerability categories
  - SQL Injection (CWE-89)
  - Command Injection (CWE-78)
  - XSS (CWE-79)
  - Hardcoded Secrets (CWE-798)
  - Path Traversal (CWE-22)
  - SSRF (CWE-918)
  - XXE (CWE-611)
  - Log Injection (CWE-117)
  - Insecure Deserialization (CWE-502)
- **MCP Integration** - Works with Claude, Cursor, Windsurf
- GitHub Actions CI/CD
- MIT License

[Unreleased]: https://github.com/mcp-tool-shop-org/brain-dev/compare/v1.0.2...HEAD
[1.0.2]: https://github.com/mcp-tool-shop-org/brain-dev/compare/v1.0.0...v1.0.2
[1.0.0]: https://github.com/mcp-tool-shop-org/brain-dev/releases/tag/v1.0.0
