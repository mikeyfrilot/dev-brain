"""
Tests for smart_test_generator module.

Tests the AST-based code analysis and test generation.
"""

import pytest
from pathlib import Path

from brain_dev.smart_test_generator import (
    Parameter,
    FunctionInfo,
    ClassInfo,
    ImportInfo,
    ModuleInfo,
    CodeAnalyzer,
    MockDetector,
    TestGenerator,
    generate_tests_for_file,
)


# ==========================================================================
# Test Data Classes
# ==========================================================================

class TestParameter:
    """Tests for Parameter dataclass."""

    def test_create_parameter(self):
        """Test creating a parameter."""
        param = Parameter(name="x", annotation="int", default="0")
        assert param.name == "x"
        assert param.annotation == "int"
        assert param.default == "0"

    def test_parameter_defaults(self):
        """Test parameter default values."""
        param = Parameter(name="x")
        assert param.annotation is None
        assert param.default is None
        assert param.is_self is False
        assert param.is_cls is False

    def test_self_parameter(self):
        """Test self parameter detection."""
        param = Parameter(name="self", is_self=True)
        assert param.is_self is True

    def test_cls_parameter(self):
        """Test cls parameter detection."""
        param = Parameter(name="cls", is_cls=True)
        assert param.is_cls is True


class TestFunctionInfo:
    """Tests for FunctionInfo dataclass."""

    def test_create_function_info(self):
        """Test creating function info."""
        func = FunctionInfo(name="my_func", is_async=True)
        assert func.name == "my_func"
        assert func.is_async is True
        assert func.params == []

    def test_function_with_params(self):
        """Test function with parameters."""
        params = [
            Parameter(name="self", is_self=True),
            Parameter(name="x", annotation="int"),
        ]
        func = FunctionInfo(name="method", params=params, is_method=True)
        assert len(func.params) == 2
        assert func.is_method is True

    def test_function_decorators(self):
        """Test function with decorators."""
        func = FunctionInfo(
            name="my_property",
            decorators=["property"],
            is_property=True,
        )
        assert "property" in func.decorators
        assert func.is_property is True


class TestClassInfo:
    """Tests for ClassInfo dataclass."""

    def test_create_class_info(self):
        """Test creating class info."""
        cls = ClassInfo(name="MyClass")
        assert cls.name == "MyClass"
        assert cls.bases == []
        assert cls.methods == []

    def test_dataclass_detection(self):
        """Test dataclass detection."""
        cls = ClassInfo(name="Data", decorators=["dataclass"], is_dataclass=True)
        assert cls.is_dataclass is True


class TestImportInfo:
    """Tests for ImportInfo dataclass."""

    def test_regular_import(self):
        """Test regular import."""
        imp = ImportInfo(module="os", is_from_import=False)
        assert imp.module == "os"
        assert imp.is_from_import is False

    def test_from_import(self):
        """Test from import."""
        imp = ImportInfo(module="pathlib", names=["Path"], is_from_import=True)
        assert imp.module == "pathlib"
        assert "Path" in imp.names
        assert imp.is_from_import is True


# ==========================================================================
# Test CodeAnalyzer
# ==========================================================================

class TestCodeAnalyzer:
    """Tests for CodeAnalyzer."""

    def test_analyze_simple_function(self):
        """Test analyzing a simple function."""
        source = '''
def hello(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}"
'''
        analyzer = CodeAnalyzer(source, "test.py")
        result = analyzer.analyze()

        assert len(result.functions) == 1
        func = result.functions[0]
        assert func.name == "hello"
        assert func.return_annotation == "str"
        assert len(func.params) == 1
        assert func.params[0].name == "name"
        assert func.params[0].annotation == "str"

    def test_analyze_async_function(self):
        """Test analyzing async function."""
        source = '''
async def fetch_data(url: str) -> dict:
    pass
'''
        analyzer = CodeAnalyzer(source, "test.py")
        result = analyzer.analyze()

        assert len(result.functions) == 1
        assert result.functions[0].is_async is True

    def test_analyze_class(self):
        """Test analyzing a class."""
        source = '''
class MyClass:
    """A test class."""

    def __init__(self, value: int):
        self.value = value

    def get_value(self) -> int:
        return self.value
'''
        analyzer = CodeAnalyzer(source, "test.py")
        result = analyzer.analyze()

        assert len(result.classes) == 1
        cls = result.classes[0]
        assert cls.name == "MyClass"
        assert len(cls.methods) == 2
        assert cls.methods[0].name == "__init__"
        assert cls.methods[1].name == "get_value"

    def test_analyze_imports(self):
        """Test analyzing imports."""
        source = '''
import os
from pathlib import Path
from typing import Optional, List
'''
        analyzer = CodeAnalyzer(source, "test.py")
        result = analyzer.analyze()

        assert len(result.imports) == 3
        assert result.imports[0].module == "os"
        assert result.imports[1].module == "pathlib"
        assert "Path" in result.imports[1].names

    def test_analyze_decorated_function(self):
        """Test analyzing decorated function."""
        source = '''
@property
def name(self) -> str:
    return self._name
'''
        analyzer = CodeAnalyzer(source, "test.py")
        result = analyzer.analyze()

        assert len(result.functions) == 1
        assert "property" in result.functions[0].decorators

    def test_analyze_function_with_raises(self):
        """Test detecting raises in function."""
        source = '''
def divide(a: int, b: int) -> float:
    """Divide a by b.

    Raises:
        ValueError: If b is zero.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
'''
        analyzer = CodeAnalyzer(source, "test.py")
        result = analyzer.analyze()

        assert len(result.functions) == 1
        assert "ValueError" in result.functions[0].raises

    def test_analyze_dataclass(self):
        """Test analyzing a dataclass."""
        source = '''
from dataclasses import dataclass

@dataclass
class Point:
    x: int
    y: int
'''
        analyzer = CodeAnalyzer(source, "test.py")
        result = analyzer.analyze()

        assert len(result.classes) == 1
        assert result.classes[0].is_dataclass is True
        assert "x" in result.classes[0].class_vars
        assert "y" in result.classes[0].class_vars

    def test_analyze_union_type(self):
        """Test analyzing union type annotations."""
        source = '''
def maybe_int(value: str) -> int | None:
    pass
'''
        analyzer = CodeAnalyzer(source, "test.py")
        result = analyzer.analyze()

        assert result.functions[0].return_annotation == "int | None"

    def test_analyze_optional_params(self):
        """Test analyzing optional parameters."""
        source = '''
def greet(name: str = "World") -> str:
    return f"Hello, {name}"
'''
        analyzer = CodeAnalyzer(source, "test.py")
        result = analyzer.analyze()

        assert result.functions[0].params[0].default == "'World'"

    def test_analyze_varargs(self):
        """Test analyzing *args and **kwargs."""
        source = '''
def func(*args, **kwargs):
    pass
'''
        analyzer = CodeAnalyzer(source, "test.py")
        result = analyzer.analyze()

        params = result.functions[0].params
        assert any("*args" in p.name for p in params)
        assert any("**kwargs" in p.name for p in params)

    def test_annotation_handler_dispatch_optional(self):
        """Regression: _ANNOTATION_HANDLERS dispatch must not raise TypeError.

        Previously the dispatch table stored unbound methods, causing
        'TypeError: missing 1 required positional argument' at runtime.
        """
        source = '''
from typing import Optional

def maybe_int(value: str) -> Optional[int]:
    """Return int or None."""
    try:
        return int(value)
    except ValueError:
        return None
'''
        analyzer = CodeAnalyzer(source, "test.py")
        result = analyzer.analyze()

        func = result.functions[0]
        assert func.return_annotation == "Optional[int]"
        assert func.params[0].annotation == "str"

    def test_annotation_handler_dispatch_generic_subscripts(self):
        """Regression: complex subscript annotations like list[str],
        dict[str, int] must be parsed without TypeError."""
        source = '''
def process(items: list[str], mapping: dict[str, int]) -> list[int]:
    pass
'''
        analyzer = CodeAnalyzer(source, "test.py")
        result = analyzer.analyze()

        func = result.functions[0]
        assert func.return_annotation == "list[int]"
        assert func.params[0].annotation == "list[str]"
        assert func.params[1].annotation == "dict[str, int]"


# ==========================================================================
# Test MockDetector
# ==========================================================================

class TestMockDetector:
    """Tests for MockDetector."""

    def test_detect_requests_mock(self):
        """Test detecting requests import needs mocking."""
        module = ModuleInfo(
            file_path="test.py",
            module_name="test",
            imports=[ImportInfo(module="requests", is_from_import=False)],
        )
        detector = MockDetector(module)
        mocks = detector.detect_mocks()

        assert "requests" in mocks

    def test_detect_httpx_async_mock(self):
        """Test detecting async httpx needs AsyncMock."""
        module = ModuleInfo(
            file_path="test.py",
            module_name="test",
            imports=[ImportInfo(module="aiohttp", is_from_import=False)],
        )
        detector = MockDetector(module)
        mocks = detector.detect_mocks()

        assert "aiohttp" in mocks
        assert mocks["aiohttp"] == "AsyncMock"

    def test_no_mock_for_stdlib(self):
        """Test that standard library utilities don't need mocking."""
        module = ModuleInfo(
            file_path="test.py",
            module_name="test",
            imports=[
                ImportInfo(module="os", is_from_import=False),
                ImportInfo(module="json", is_from_import=False),
            ],
        )
        detector = MockDetector(module)
        mocks = detector.detect_mocks()

        assert "os" not in mocks
        assert "json" not in mocks


# ==========================================================================
# Test TestGenerator
# ==========================================================================

class TestTestGeneratorClass:
    """Tests for TestGenerator class."""

    @pytest.fixture
    def simple_module(self):
        """Create a simple module for testing."""
        return ModuleInfo(
            file_path="mymodule.py",
            module_name="mymodule",
            imports=[],
            functions=[
                FunctionInfo(
                    name="add",
                    params=[
                        Parameter(name="a", annotation="int"),
                        Parameter(name="b", annotation="int"),
                    ],
                    return_annotation="int",
                ),
            ],
            classes=[],
        )

    def test_generate_header(self, simple_module):
        """Test header generation."""
        generator = TestGenerator(simple_module, {})
        result = generator.generate_test_file()

        assert "Tests for mymodule" in result
        assert "Auto-generated by Dev Brain" in result

    def test_generate_imports(self, simple_module):
        """Test import generation."""
        generator = TestGenerator(simple_module, {})
        result = generator.generate_test_file()

        assert "import pytest" in result
        assert "from mymodule import" in result

    def test_generate_function_test(self, simple_module):
        """Test function test generation."""
        generator = TestGenerator(simple_module, {})
        result = generator.generate_test_file()

        assert "def test_add" in result
        assert "result = add(" in result
        assert "assert" in result

    def test_generate_async_test(self):
        """Test async function test generation."""
        module = ModuleInfo(
            file_path="async_mod.py",
            module_name="async_mod",
            imports=[],
            functions=[
                FunctionInfo(
                    name="fetch",
                    params=[Parameter(name="url", annotation="str")],
                    return_annotation="dict",
                    is_async=True,
                ),
            ],
            classes=[],
        )
        generator = TestGenerator(module, {})
        result = generator.generate_test_file()

        assert "@pytest.mark.asyncio" in result
        assert "async def test_fetch" in result
        assert "await fetch(" in result

    def test_generate_class_tests(self):
        """Test class test generation."""
        module = ModuleInfo(
            file_path="cls_mod.py",
            module_name="cls_mod",
            imports=[],
            functions=[],
            classes=[
                ClassInfo(
                    name="Calculator",
                    methods=[
                        FunctionInfo(
                            name="__init__",
                            params=[Parameter(name="self", is_self=True)],
                            is_method=True,
                        ),
                        FunctionInfo(
                            name="add",
                            params=[
                                Parameter(name="self", is_self=True),
                                Parameter(name="a", annotation="int"),
                                Parameter(name="b", annotation="int"),
                            ],
                            return_annotation="int",
                            is_method=True,
                        ),
                    ],
                ),
            ],
        )
        generator = TestGenerator(module, {})
        result = generator.generate_test_file()

        assert "class TestCalculator:" in result
        assert "def test_init" in result
        assert "def test_add" in result

    def test_generate_fixtures_for_mocks(self):
        """Test fixture generation for mocks."""
        module = ModuleInfo(
            file_path="test.py",
            module_name="test",
            imports=[],
            functions=[],
            classes=[],
        )
        mocks = {"requests": "MagicMock", "aiohttp": "AsyncMock"}
        generator = TestGenerator(module, mocks)
        result = generator.generate_test_file()

        assert "@pytest.fixture" in result
        assert "mock_requests" in result
        assert "mock_aiohttp" in result


# ==========================================================================
# Test generate_tests_for_file
# ==========================================================================

class TestGenerateTestsForFile:
    """Tests for the main entry point."""

    def test_generate_for_real_file(self, tmp_path):
        """Test generating tests for a real file."""
        # Create a temp Python file
        test_file = tmp_path / "sample.py"
        test_file.write_text('''
def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}"

class Greeter:
    def __init__(self, prefix: str = "Hello"):
        self.prefix = prefix

    def greet(self, name: str) -> str:
        return f"{self.prefix}, {name}"
''')

        result = generate_tests_for_file(str(test_file))

        assert "Tests for" in result
        assert "def test_greet" in result
        assert "class TestGreeter" in result
        assert "def test_init" in result

    def test_generate_preserves_async(self, tmp_path):
        """Test that async functions get async tests."""
        test_file = tmp_path / "async_sample.py"
        test_file.write_text('''
async def fetch_data(url: str) -> dict:
    """Fetch data from URL."""
    pass
''')

        result = generate_tests_for_file(str(test_file))

        assert "@pytest.mark.asyncio" in result
        assert "async def test_fetch_data" in result


# ==========================================================================
# Test Edge Cases
# ==========================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_file(self, tmp_path):
        """Test generating tests for empty file."""
        test_file = tmp_path / "empty.py"
        test_file.write_text("")

        result = generate_tests_for_file(str(test_file))

        assert "Tests for" in result

    def test_private_functions_skipped(self, tmp_path):
        """Test that private functions are skipped."""
        test_file = tmp_path / "private.py"
        test_file.write_text('''
def _private_helper():
    pass

def public_func():
    pass
''')

        result = generate_tests_for_file(str(test_file))

        assert "test__private_helper" not in result
        assert "test_public_func" in result

    def test_complex_annotations(self, tmp_path):
        """Test handling complex type annotations."""
        test_file = tmp_path / "complex.py"
        test_file.write_text('''
from typing import Optional, List, Dict

def process(
    items: List[Dict[str, int]],
    default: Optional[int] = None,
) -> Dict[str, List[int]]:
    pass
''')

        result = generate_tests_for_file(str(test_file))

        assert "def test_process" in result

    def test_optional_return_assertion_not_tautological(self, tmp_path):
        """Regression: generated tests for Optional[T] returns must contain
        a real isinstance check, not the tautology 'result is None or result is not None'.
        """
        test_file = tmp_path / "opt_return.py"
        test_file.write_text('''
from typing import Optional

def find_user(user_id: str) -> Optional[int]:
    """Look up a user by ID."""
    return 42
''')

        result = generate_tests_for_file(str(test_file))

        # Must NOT contain the tautology
        assert "result is None or result is not None" not in result
        # Must contain a meaningful isinstance check
        assert "isinstance(result, int)" in result

    def test_optional_return_assertion_for_str(self, tmp_path):
        """Regression: Optional[str] should get isinstance(result, str)."""
        test_file = tmp_path / "opt_str.py"
        test_file.write_text('''
from typing import Optional

def get_name(key: str) -> Optional[str]:
    return "hello"
''')

        result = generate_tests_for_file(str(test_file))

        assert "isinstance(result, str)" in result
        assert "result is None or result is not None" not in result
