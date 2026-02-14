"""
Smart Test Generator - AST-based automatic test generation.

2026 Best Practices:
- Uses ast.NodeVisitor for accurate code parsing
- Detects dependencies that need mocking
- Generates complete pytest tests with fixtures
- Supports async functions
- Type-aware test generation

References:
- https://docs.python.org/3/library/ast.html
- https://pyanalyze.readthedocs.io/en/latest/design.html
- https://mcp.so/server/python-testing/jazzberry-ai
"""

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class Parameter:
    """Function parameter info."""
    name: str
    annotation: Optional[str] = None
    default: Optional[str] = None
    is_self: bool = False
    is_cls: bool = False


@dataclass
class FunctionInfo:
    """Extracted function information."""
    name: str
    params: list[Parameter] = field(default_factory=list)
    return_annotation: Optional[str] = None
    is_async: bool = False
    is_method: bool = False
    is_classmethod: bool = False
    is_staticmethod: bool = False
    is_property: bool = False
    decorators: list[str] = field(default_factory=list)
    docstring: Optional[str] = None
    raises: list[str] = field(default_factory=list)
    line_number: int = 0
    class_name: Optional[str] = None


@dataclass
class ClassInfo:
    """Extracted class information."""
    name: str
    bases: list[str] = field(default_factory=list)
    methods: list[FunctionInfo] = field(default_factory=list)
    class_vars: list[str] = field(default_factory=list)
    docstring: Optional[str] = None
    line_number: int = 0
    is_dataclass: bool = False
    decorators: list[str] = field(default_factory=list)


@dataclass
class ImportInfo:
    """Import information for mock detection."""
    module: str
    names: list[str] = field(default_factory=list)
    alias: Optional[str] = None
    is_from_import: bool = False


@dataclass
class ModuleInfo:
    """Complete module analysis."""
    file_path: str
    module_name: str
    imports: list[ImportInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    functions: list[FunctionInfo] = field(default_factory=list)
    global_vars: list[str] = field(default_factory=list)


class CodeAnalyzer(ast.NodeVisitor):
    """
    AST-based code analyzer for test generation.

    Extracts:
    - Function signatures with types
    - Class definitions with methods
    - Import statements for mock detection
    - Docstrings and decorators
    """

    def __init__(self, source_code: str, file_path: str = ""):
        self.source_code = source_code
        self.file_path = file_path
        self.module_name = self._detect_module_path(file_path) if file_path else "module"

        # Results
        self.imports: list[ImportInfo] = []
        self.classes: list[ClassInfo] = []
        self.functions: list[FunctionInfo] = []
        self.global_vars: list[str] = []

        # Context tracking
        self._current_class: Optional[ClassInfo] = None

    def _detect_module_path(self, file_path: str) -> str:
        """Detect full module path by walking up to find package root."""
        path = Path(file_path).resolve()
        parts = [path.stem]  # Start with module name (without .py)

        # Walk up looking for __init__.py files
        parent = path.parent
        while parent != parent.parent:  # Stop at root
            init_file = parent / "__init__.py"
            if init_file.exists():
                parts.insert(0, parent.name)
                parent = parent.parent
            else:
                break

        return ".".join(parts)

    def analyze(self) -> ModuleInfo:
        """Analyze the source code and return module info."""
        tree = ast.parse(self.source_code)
        self.visit(tree)

        return ModuleInfo(
            file_path=self.file_path,
            module_name=self.module_name,
            imports=self.imports,
            classes=self.classes,
            functions=self.functions,
            global_vars=self.global_vars,
        )

    def visit_Import(self, node: ast.Import) -> None:
        """Handle import statements."""
        for alias in node.names:
            self.imports.append(ImportInfo(
                module=alias.name,
                alias=alias.asname,
                is_from_import=False,
            ))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Handle from ... import statements."""
        module = node.module or ""
        names = [alias.name for alias in node.names]
        self.imports.append(ImportInfo(
            module=module,
            names=names,
            is_from_import=True,
        ))
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Handle class definitions."""
        # Get decorators
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]

        # Get base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(f"{self._get_attribute_path(base)}")

        # Create class info
        class_info = ClassInfo(
            name=node.name,
            bases=bases,
            decorators=decorators,
            docstring=ast.get_docstring(node),
            line_number=node.lineno,
            is_dataclass="dataclass" in decorators,
        )

        # Get class variables
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                class_info.class_vars.append(item.target.id)
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        class_info.class_vars.append(target.id)

        # Process methods
        self._current_class = class_info
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.visit(item)
        self._current_class = None

        self.classes.append(class_info)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Handle function definitions."""
        self._process_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Handle async function definitions."""
        self._process_function(node, is_async=True)

    def _process_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        is_async: bool
    ) -> None:
        """Process a function or method definition."""
        # Get decorators
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]

        # Parse parameters
        params = self._parse_parameters(node.args)

        # Get return annotation
        return_annotation = None
        if node.returns:
            return_annotation = self._annotation_to_string(node.returns)

        # Detect raises from docstring or body
        raises = self._detect_raises(node)

        func_info = FunctionInfo(
            name=node.name,
            params=params,
            return_annotation=return_annotation,
            is_async=is_async,
            is_method=self._current_class is not None,
            is_classmethod="classmethod" in decorators,
            is_staticmethod="staticmethod" in decorators,
            is_property="property" in decorators,
            decorators=decorators,
            docstring=ast.get_docstring(node),
            raises=raises,
            line_number=node.lineno,
            class_name=self._current_class.name if self._current_class else None,
        )

        if self._current_class:
            self._current_class.methods.append(func_info)
        else:
            self.functions.append(func_info)

    def _parse_parameters(self, args: ast.arguments) -> list[Parameter]:
        """Parse function parameters."""
        params = []

        # Regular args
        defaults_offset = len(args.args) - len(args.defaults)
        for i, arg in enumerate(args.args):
            default = None
            if i >= defaults_offset:
                default_node = args.defaults[i - defaults_offset]
                default = self._get_default_value(default_node)

            params.append(Parameter(
                name=arg.arg,
                annotation=self._annotation_to_string(arg.annotation) if arg.annotation else None,
                default=default,
                is_self=arg.arg == "self",
                is_cls=arg.arg == "cls",
            ))

        # *args
        if args.vararg:
            params.append(Parameter(
                name=f"*{args.vararg.arg}",
                annotation=self._annotation_to_string(args.vararg.annotation) if args.vararg.annotation else None,
            ))

        # **kwargs
        if args.kwarg:
            params.append(Parameter(
                name=f"**{args.kwarg.arg}",
                annotation=self._annotation_to_string(args.kwarg.annotation) if args.kwarg.annotation else None,
            ))

        return params

    def _annotation_to_string(self, node: ast.expr | None) -> Optional[str]:
        """Convert annotation AST node to string."""
        if node is None:
            return None

        # Dispatch to type-specific handlers
        handler_name = self._ANNOTATION_HANDLERS.get(type(node).__name__)
        if handler_name:
            method = getattr(self, handler_name)
            return method(node)

        return ast.unparse(node) if hasattr(ast, 'unparse') else "Any"

    def _handle_name_annotation(self, node: ast.Name) -> str:
        return node.id

    def _handle_constant_annotation(self, node: ast.Constant) -> str:
        return repr(node.value)

    def _handle_subscript_annotation(self, node: ast.Subscript) -> str:
        value = self._annotation_to_string(node.value)
        slice_val = self._annotation_to_string(node.slice)
        return f"{value}[{slice_val}]"

    def _handle_attribute_annotation(self, node: ast.Attribute) -> str:
        return self._get_attribute_path(node)

    def _handle_tuple_annotation(self, node: ast.Tuple) -> str:
        elements = [self._annotation_to_string(e) for e in node.elts]
        return ", ".join(e for e in elements if e)

    def _handle_binop_annotation(self, node: ast.BinOp) -> Optional[str]:
        if isinstance(node.op, ast.BitOr):
            left = self._annotation_to_string(node.left)
            right = self._annotation_to_string(node.right)
            return f"{left} | {right}"
        return None

    # Dispatch table for annotation handling (reduces cyclomatic complexity).
    # Maps AST node type names to handler method names (strings) so that
    # getattr(self, name) can bind them correctly at call time.
    _ANNOTATION_HANDLERS: dict[str, str] = {
        "Name": "_handle_name_annotation",
        "Constant": "_handle_constant_annotation",
        "Subscript": "_handle_subscript_annotation",
        "Attribute": "_handle_attribute_annotation",
        "Tuple": "_handle_tuple_annotation",
        "BinOp": "_handle_binop_annotation",
    }

    def _get_decorator_name(self, node: ast.expr) -> str:
        """Get decorator name from node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._get_attribute_path(node)
        elif isinstance(node, ast.Call):
            return self._get_decorator_name(node.func)
        return "unknown"

    def _get_attribute_path(self, node: ast.Attribute) -> str:
        """Get full attribute path (e.g., 'module.Class.method')."""
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))

    def _get_default_value(self, node: ast.expr) -> str:
        """Get string representation of default value."""
        if isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.List):
            return "[]"
        elif isinstance(node, ast.Dict):
            return "{}"
        elif isinstance(node, ast.Call):
            func = self._get_decorator_name(node.func)
            return f"{func}()"
        return "..."

    def _detect_raises(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
        """Detect exceptions raised by a function."""
        raises = []

        # Check docstring for :raises: or Raises: sections
        docstring = ast.get_docstring(node) or ""
        raises_pattern = r'(?:raises?|Raises?)[:\s]+(\w+)'
        raises.extend(re.findall(raises_pattern, docstring))

        # Check body for raise statements
        for child in ast.walk(node):
            if isinstance(child, ast.Raise) and child.exc:
                if isinstance(child.exc, ast.Call) and isinstance(child.exc.func, ast.Name):
                    raises.append(child.exc.func.id)
                elif isinstance(child.exc, ast.Name):
                    raises.append(child.exc.id)

        return list(set(raises))


class MockDetector:
    """
    Detects what needs to be mocked based on imports and usage.
    """

    # Common patterns that typically need mocking
    MOCK_PATTERNS = {
        # External services
        "requests", "httpx", "aiohttp", "urllib",
        # Databases
        "sqlalchemy", "pymongo", "redis", "psycopg",
        # File I/O
        "open", "pathlib.Path",
        # Time/Random
        "time.time", "datetime.now", "random",
        # Async
        "asyncio.sleep",
        # Third party
        "gradio", "websockets",
    }

    # Known async libraries
    ASYNC_LIBS = {"asyncio", "aiohttp", "websockets", "aiofiles"}

    def __init__(self, module_info: ModuleInfo):
        self.module_info = module_info
        self.mock_suggestions: dict[str, str] = {}

    def detect_mocks(self) -> dict[str, str]:
        """
        Detect what needs mocking and return suggestions.

        Returns:
            Dict mapping import name to mock suggestion
        """
        for imp in self.module_info.imports:
            if imp.is_from_import:
                # from x import y, z
                for name in imp.names:
                    full_path = f"{imp.module}.{name}"
                    if self._should_mock(full_path, imp.module):
                        self.mock_suggestions[name] = self._get_mock_type(full_path)
            else:
                # import x
                if self._should_mock(imp.module, imp.module):
                    name = imp.alias or imp.module.split(".")[0]
                    self.mock_suggestions[name] = self._get_mock_type(imp.module)

        return self.mock_suggestions

    def _should_mock(self, path: str, module: str) -> bool:
        """Check if this import should be mocked."""
        # Check against known patterns
        for pattern in self.MOCK_PATTERNS:
            if pattern in path or pattern in module:
                return True

        # External packages (not from same module)
        if not module.startswith(self.module_info.module_name.split(".")[0]):
            # Check if it's a well-known external package
            top_level = module.split(".")[0]
            if top_level in {"os", "sys", "json", "re", "typing", "dataclasses"}:
                return False  # Standard library utilities, usually no need to mock
            return True

        return False

    def _get_mock_type(self, path: str) -> str:
        """Get the appropriate mock type for a path."""
        # Check if async
        for async_lib in self.ASYNC_LIBS:
            if async_lib in path:
                return "AsyncMock"

        # Default to MagicMock
        return "MagicMock"


class SmartPytestFileGenerator:
    """
    Generates complete pytest test files.

    Note: Named SmartPytestFileGenerator to avoid pytest collection warnings
    (pytest collects classes with "Test" prefix OR suffix).
    """

    def __init__(self, module_info: ModuleInfo, mocks: dict[str, str]):
        self.module_info = module_info
        self.mocks = mocks

    def generate_test_file(self) -> str:
        """Generate a complete test file."""
        parts = []

        # Header
        parts.append(self._generate_header())

        # Imports
        parts.append(self._generate_imports())

        # Fixtures
        fixtures = self._generate_fixtures()
        if fixtures:
            parts.append(fixtures)

        # Tests for module-level functions
        for func in self.module_info.functions:
            if not func.name.startswith("_"):  # Skip private
                parts.append(self._generate_function_tests(func))

        # Tests for classes
        for cls in self.module_info.classes:
            parts.append(self._generate_class_tests(cls))

        return "\n\n".join(parts)

    def _generate_header(self) -> str:
        """Generate file header with docstring."""
        return f'''"""
Tests for {self.module_info.module_name}.

Auto-generated by Dev Brain Smart Test Generator.
"""'''

    def _generate_imports(self) -> str:
        """Generate import statements."""
        lines = [
            "import pytest",
            "from unittest.mock import MagicMock, AsyncMock, patch",
        ]

        # Check if we need asyncio
        has_async = any(
            f.is_async for f in self.module_info.functions
        ) or any(
            any(m.is_async for m in c.methods) for c in self.module_info.classes
        )

        if has_async:
            lines.append("import asyncio")

        # Import the module under test
        module_path = self.module_info.file_path.replace("/", ".").replace("\\", ".")
        module_path = module_path.replace(".py", "")

        # Build import statement
        classes = [c.name for c in self.module_info.classes]
        functions = [f.name for f in self.module_info.functions if not f.name.startswith("_")]

        items_to_import = classes + functions
        if items_to_import:
            # Use relative path style import
            lines.append(f"\nfrom {self.module_info.module_name} import (")
            for item in items_to_import:
                lines.append(f"    {item},")
            lines.append(")")

        return "\n".join(lines)

    def _generate_fixtures(self) -> str:
        """Generate pytest fixtures for mocks."""
        if not self.mocks:
            return ""

        lines = ["\n# " + "=" * 70, "# Fixtures", "# " + "=" * 70]

        for name, mock_type in self.mocks.items():
            fixture_name = f"mock_{name.lower()}"
            lines.append(f'''
@pytest.fixture
def {fixture_name}():
    """Mock for {name}."""
    return {mock_type}()
''')

        return "\n".join(lines)

    def _generate_function_tests(self, func: FunctionInfo) -> str:
        """Generate tests for a standalone function."""
        lines = [
            f"\n# " + "=" * 70,
            f"# Tests for {func.name}",
            f"# " + "=" * 70,
        ]

        # Generate basic test
        test_name = f"test_{func.name}"
        if func.is_async:
            lines.append(self._generate_async_test(func, test_name))
        else:
            lines.append(self._generate_sync_test(func, test_name))

        # Generate edge case tests
        if func.params:
            lines.append(self._generate_edge_case_test(func))

        # Generate error tests if function raises
        if func.raises:
            lines.append(self._generate_error_test(func))

        return "\n".join(lines)

    def _generate_class_tests(self, cls: ClassInfo) -> str:
        """Generate tests for a class."""
        lines = [
            f"\n# " + "=" * 70,
            f"# Tests for {cls.name}",
            f"# " + "=" * 70,
        ]

        # Test class with fixtures
        lines.append(f"\nclass Test{cls.name}:")
        lines.append(f'    """Tests for the {cls.name} class."""')

        # Fixture for the class instance
        lines.append(self._generate_class_fixture(cls))

        # Test __init__ if exists
        init_method = next((m for m in cls.methods if m.name == "__init__"), None)
        if init_method:
            lines.append(self._generate_init_test(cls, init_method))

        # Test each public method
        for method in cls.methods:
            if not method.name.startswith("_") and method.name != "__init__":
                lines.append(self._generate_method_test(cls, method))

        return "\n".join(lines)

    def _generate_class_fixture(self, cls: ClassInfo) -> str:
        """Generate fixture for class instance."""
        # Find __init__ params
        init_method = next((m for m in cls.methods if m.name == "__init__"), None)

        params = []
        if init_method:
            for p in init_method.params:
                if not p.is_self and not p.is_cls:
                    if p.default:
                        continue  # Skip params with defaults
                    params.append(self._get_mock_value(p))

        params_str = ", ".join(params) if params else ""

        return f'''
    @pytest.fixture
    def instance(self):
        """Create a {cls.name} instance for testing."""
        return {cls.name}({params_str})
'''

    def _generate_init_test(self, cls: ClassInfo, init: FunctionInfo) -> str:
        """Generate test for __init__ method."""
        params = [p for p in init.params if not p.is_self]

        if not params:
            return f'''
    def test_init(self):
        """Test {cls.name} initialization."""
        instance = {cls.name}()
        assert instance is not None
'''

        # Generate with mock values
        setup_lines = []
        call_params = []

        for p in params:
            if p.default is None:
                mock_val = self._get_mock_value(p)
                setup_lines.append(f"        {p.name} = {mock_val}")
                call_params.append(f"{p.name}={p.name}")

        setup = "\n".join(setup_lines)
        call = ", ".join(call_params)

        return f'''
    def test_init(self):
        """Test {cls.name} initialization."""
{setup}

        instance = {cls.name}({call})

        assert instance is not None
'''

    def _generate_method_test(self, cls: ClassInfo, method: FunctionInfo) -> str:
        """Generate test for a class method."""
        test_name = f"test_{method.name}"

        if method.is_async:
            return self._generate_async_method_test(cls, method, test_name)
        else:
            return self._generate_sync_method_test(cls, method, test_name)

    def _generate_sync_test(self, func: FunctionInfo, test_name: str) -> str:
        """Generate a synchronous test."""
        # Filter out self/cls and *args/**kwargs
        params = [
            p for p in func.params
            if not p.is_self and not p.is_cls
            and not p.name.startswith("*")
        ]

        # Setup
        setup_lines = []
        call_params = []
        for p in params:
            if p.default is None:
                mock_val = self._get_mock_value(p)
                setup_lines.append(f"    {p.name} = {mock_val}")
                call_params.append(p.name)

        setup = "\n".join(setup_lines) if setup_lines else "    # No setup needed"
        call = ", ".join(call_params)

        # Assertion based on return type
        assertion = self._get_assertion(func)

        return f'''
def {test_name}():
    """Test {func.name} basic functionality."""
{setup}

    result = {func.name}({call})

{assertion}
'''

    def _generate_async_test(self, func: FunctionInfo, test_name: str) -> str:
        """Generate an async test."""
        # Filter out self/cls and *args/**kwargs
        params = [
            p for p in func.params
            if not p.is_self and not p.is_cls
            and not p.name.startswith("*")
        ]

        # Setup
        setup_lines = []
        call_params = []
        for p in params:
            if p.default is None:
                mock_val = self._get_mock_value(p)
                setup_lines.append(f"    {p.name} = {mock_val}")
                call_params.append(p.name)

        setup = "\n".join(setup_lines) if setup_lines else "    # No setup needed"
        call = ", ".join(call_params)
        assertion = self._get_assertion(func)

        return f'''
@pytest.mark.asyncio
async def {test_name}():
    """Test {func.name} basic functionality."""
{setup}

    result = await {func.name}({call})

{assertion}
'''

    def _generate_sync_method_test(
        self, cls: ClassInfo, method: FunctionInfo, test_name: str
    ) -> str:
        """Generate a sync method test."""
        # Filter out self/cls and *args/**kwargs
        params = [
            p for p in method.params
            if not p.is_self and not p.is_cls
            and not p.name.startswith("*")
        ]

        setup_lines = []
        call_params = []
        for p in params:
            if p.default is None:
                mock_val = self._get_mock_value(p)
                setup_lines.append(f"        {p.name} = {mock_val}")
                call_params.append(p.name)

        setup = "\n".join(setup_lines) if setup_lines else "        # No setup needed"
        call = ", ".join(call_params)
        assertion = self._get_assertion(method, indent=8)

        return f'''
    def {test_name}(self, instance):
        """Test {method.name} method."""
{setup}

        result = instance.{method.name}({call})

{assertion}
'''

    def _generate_async_method_test(
        self, cls: ClassInfo, method: FunctionInfo, test_name: str
    ) -> str:
        """Generate an async method test."""
        # Filter out self/cls and *args/**kwargs
        params = [
            p for p in method.params
            if not p.is_self and not p.is_cls
            and not p.name.startswith("*")
        ]

        setup_lines = []
        call_params = []
        for p in params:
            if p.default is None:
                mock_val = self._get_mock_value(p)
                setup_lines.append(f"        {p.name} = {mock_val}")
                call_params.append(p.name)

        setup = "\n".join(setup_lines) if setup_lines else "        # No setup needed"
        call = ", ".join(call_params)
        assertion = self._get_assertion(method, indent=8)

        return f'''
    @pytest.mark.asyncio
    async def {test_name}(self, instance):
        """Test {method.name} method."""
{setup}

        result = await instance.{method.name}({call})

{assertion}
'''

    def _generate_edge_case_test(self, func: FunctionInfo) -> str:
        """Generate edge case tests."""
        test_name = f"test_{func.name}_with_none"

        # Find first optional param
        optional_params = [p for p in func.params if "Optional" in (p.annotation or "")]

        if not optional_params:
            return ""

        param = optional_params[0]

        # Build a meaningful assertion for the return type
        ret = func.return_annotation or ""
        assertion = "assert result is not None or result is None  # TODO: add type check"
        if "Optional" in ret:
            inner = self._extract_optional_inner(ret)
            if inner:
                runtime_type = self._OPTIONAL_INNER_TYPES.get(inner)
                if runtime_type:
                    assertion = f"assert result is None or isinstance(result, {runtime_type})"
        elif ret and ret in self._ASSERTION_MAP:
            assertion = self._ASSERTION_MAP[ret]

        if func.is_async:
            return f'''
@pytest.mark.asyncio
async def {test_name}():
    """Test {func.name} with None value."""
    result = await {func.name}({param.name}=None)
    {assertion}
'''
        else:
            return f'''
def {test_name}():
    """Test {func.name} with None value."""
    result = {func.name}({param.name}=None)
    {assertion}
'''

    def _generate_error_test(self, func: FunctionInfo) -> str:
        """Generate error handling tests."""
        if not func.raises:
            return ""

        exc = func.raises[0]
        test_name = f"test_{func.name}_raises_{exc.lower()}"

        if func.is_async:
            return f'''
@pytest.mark.asyncio
async def {test_name}():
    """Test {func.name} raises {exc}."""
    with pytest.raises({exc}):
        await {func.name}()  # Add invalid args to trigger error
'''
        else:
            return f'''
def {test_name}():
    """Test {func.name} raises {exc}."""
    with pytest.raises({exc}):
        {func.name}()  # Add invalid args to trigger error
'''

    def _get_mock_value(self, param: Parameter) -> str:
        """Get an appropriate mock value for a parameter."""
        annotation = param.annotation or ""

        # Type-based mock values
        type_mocks = {
            "str": '"test_value"',
            "int": "42",
            "float": "3.14",
            "bool": "True",
            "list": "[]",
            "dict": "{}",
            "None": "None",
            "Optional": "None",
            "Path": 'Path("/tmp/test")',
        }

        for type_name, mock_val in type_mocks.items():
            if type_name in annotation:
                return mock_val

        # Check param name for hints
        if "path" in param.name.lower():
            return 'Path("/tmp/test")'
        if "name" in param.name.lower():
            return '"test_name"'
        if "id" in param.name.lower():
            return '"test_id"'
        if "url" in param.name.lower():
            return '"https://example.com"'
        if "config" in param.name.lower():
            return "MagicMock()"

        # Default to MagicMock
        return "MagicMock()"

    # Dispatch table for assertion generation (reduces cyclomatic complexity)
    _ASSERTION_MAP: dict[str, str] = {
        "None": "assert result is None",
        "": "assert result is None",
        "bool": "assert isinstance(result, bool)",
        "str": "assert isinstance(result, str)",
        "int": "assert isinstance(result, int)",
        "float": "assert isinstance(result, (int, float))",
    }

    # Maps type name strings to their runtime type expression for isinstance checks
    _OPTIONAL_INNER_TYPES: dict[str, str] = {
        "int": "int",
        "str": "str",
        "float": "(int, float)",
        "bool": "bool",
        "list": "list",
        "dict": "dict",
        "tuple": "tuple",
        "set": "set",
        "bytes": "bytes",
    }

    @staticmethod
    def _extract_optional_inner(ret: str) -> Optional[str]:
        """Extract the inner type T from 'Optional[T]' annotations.

        Returns the inner type string, or None if not parseable.
        """
        # Match Optional[T] where T is the inner type
        m = re.match(r"Optional\[(.+)\]$", ret)
        if m:
            return m.group(1).strip()
        return None

    def _get_assertion(self, func: FunctionInfo, indent: int = 4) -> str:
        """Generate appropriate assertion based on return type."""
        ind = " " * indent
        ret = func.return_annotation or ""

        # Check exact matches first
        if ret in self._ASSERTION_MAP:
            return f"{ind}{self._ASSERTION_MAP[ret]}"

        # Check partial matches
        if "list" in ret.lower():
            return f"{ind}assert isinstance(result, list)"
        if "dict" in ret.lower():
            return f"{ind}assert isinstance(result, dict)"
        if "Optional" in ret:
            inner = self._extract_optional_inner(ret)
            if inner:
                # Look up the runtime type for a proper isinstance check
                runtime_type = self._OPTIONAL_INNER_TYPES.get(inner)
                if runtime_type:
                    return f"{ind}assert result is None or isinstance(result, {runtime_type})"
            return f"{ind}assert result is None or result is not None  # TODO: add type check"

        return f"{ind}assert result is not None"


def generate_tests_for_file(file_path: str) -> str:
    """
    Main entry point: Generate tests for a Python file.

    Args:
        file_path: Path to Python source file

    Returns:
        Complete test file content
    """
    with open(file_path, "r", encoding="utf-8") as f:
        source_code = f.read()

    # Analyze the code
    analyzer = CodeAnalyzer(source_code, file_path)
    module_info = analyzer.analyze()

    # Detect what needs mocking
    mock_detector = MockDetector(module_info)
    mocks = mock_detector.detect_mocks()

    # Generate tests
    generator = SmartPytestFileGenerator(module_info, mocks)
    return generator.generate_test_file()


# Backwards compatibility aliases
SmartTestFileGenerator = SmartPytestFileGenerator
TestGenerator = SmartPytestFileGenerator


# CLI interface
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python smart_test_generator.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    test_code = generate_tests_for_file(file_path)
    print(test_code)
