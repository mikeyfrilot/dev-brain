# Changelog

All notable changes to Dev Brain will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
  - `code_health_score` - Overall quality scoring
- **Security Detection** - 49 patterns across 6+ vulnerability types
  - SQL Injection (CWE-89)
  - Command Injection (CWE-78)
  - XSS (CWE-79)
  - Hardcoded Secrets (CWE-798)
  - Path Traversal (CWE-22)
  - SSRF (CWE-918)
- **Multi-Language Support**
  - Python (full AST analysis)
  - JavaScript/TypeScript
  - PowerShell
  - CSS/HTML
- **Code Health Scoring** - 11 category breakdown
- **MCP Integration** - Works with Claude, Cursor, Windsurf
- **FastMCP Framework** - Built on modern Python MCP SDK

### Infrastructure
- GitHub Actions CI/CD
- 90% test coverage
- MIT License

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 1.0.0 | 2026-01-24 | Initial release |

[Unreleased]: https://github.com/mcp-tool-shop/dev-brain/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/mcp-tool-shop/dev-brain/releases/tag/v1.0.0
