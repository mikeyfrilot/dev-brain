# Contributing to Dev Brain

Dev Brain is an MCP server that provides AI-powered code analysis tools. Contributions are welcome!

## Development Setup

```bash
git clone https://github.com/mcp-tool-shop/dev-brain.git
cd dev-brain
pip install -e ".[dev]"
pytest
```

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. Check existing [issues](https://github.com/mcp-tool-shop/dev-brain/issues)
2. If not found, create a new issue with:
   - Clear description of the problem or feature
   - Steps to reproduce (for bugs)
   - Expected vs. actual behavior
   - Your environment (Python version, OS, MCP client)

### Contributing Code

1. **Fork the repository** and create a branch from `main`
2. **Make your changes**
   - Follow existing code style
   - Use type hints
   - Add tests for new functionality
3. **Test your changes**
   ```bash
   pytest
   pytest --cov=dev_brain
   ```
4. **Commit your changes**
   - Use clear, descriptive commit messages
   - Reference issue numbers when applicable
5. **Submit a pull request**
   - Describe what your PR does and why
   - Link to related issues

## Project Structure

```
dev-brain/
├── dev_brain/
│   ├── server.py          # Main MCP server
│   ├── tools/             # Analysis tool implementations
│   │   ├── test_gen.py    # Test generation
│   │   ├── security.py    # Security scanning
│   │   ├── coverage.py    # Coverage analysis
│   │   └── refactor.py    # Refactoring suggestions
│   └── utils/             # Shared utilities
├── tests/                 # Test suite
└── spec.md               # Tool specifications
```

## Adding New Analysis Tools

When adding a new MCP tool:

1. **Define the tool** in `dev_brain/server.py`:
   ```python
   @server.call_tool()
   async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
       if name == "your_tool_name":
           return await your_tool_implementation(arguments)
   ```

2. **Implement the analysis** in `dev_brain/tools/`:
   - Use AST parsing for code analysis
   - Return structured JSON results
   - Handle errors gracefully

3. **Add tests** in `tests/`:
   - Test with real code samples
   - Test error cases
   - Test edge cases

4. **Document the tool** in `spec.md` and `README.md`

## Testing

- Write tests for new functionality
- Maintain 90%+ coverage
- Use fixtures for common test data
- Test both success and error paths

Run tests:
```bash
pytest                    # Run all tests
pytest -v                 # Verbose output
pytest --cov=dev_brain    # With coverage
pytest -k test_security   # Run specific tests
```

## Code Quality

### Type Hints
Use type hints for all functions:
```python
async def analyze_code(path: str) -> dict[str, Any]:
    ...
```

### AST Analysis
When working with Python AST:
- Use `ast.parse()` for code parsing
- Use `ast.walk()` for tree traversal
- Handle syntax errors gracefully

### MCP Protocol
Follow MCP conventions:
- Tools return `list[types.TextContent]`
- Use structured JSON for results
- Include error details in responses

## Documentation

When adding features:

- Update `README.md` with usage examples
- Add tool specifications to `spec.md`
- Include inline comments for complex logic
- Update `CHANGELOG.md`

## Security Considerations

When modifying security scanning:

- Test against known vulnerability patterns
- Avoid false positives
- Document detection methods
- Add test cases for new patterns

## Tool Categories

Dev Brain provides 9 tool categories:

1. **Test Generation** - Generate pytest files from AST
2. **Security Scanning** - Detect vulnerabilities (SQL injection, secrets, etc.)
3. **Coverage Analysis** - Find untested code paths
4. **Refactoring Suggestions** - Identify complexity and duplication
5. **Documentation Analysis** - Find missing docstrings
6. **UX Insights** - Analyze user-facing code
7. **Performance Profiling** - Identify bottlenecks
8. **Dependency Analysis** - Check for outdated/vulnerable packages
9. **Architecture Analysis** - Assess code structure

## Release Process

(For maintainers)

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create git tag: `git tag v1.x.x`
4. Push tag: `git push origin v1.x.x`
5. GitHub Actions will publish to PyPI

## Questions?

Open an issue or start a discussion in the [MCP Tool Shop](https://github.com/mcp-tool-shop) organization.
