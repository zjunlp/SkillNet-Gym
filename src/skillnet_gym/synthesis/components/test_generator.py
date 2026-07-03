"""Test generator for creating pytest tests based on trajectory outputs.

.. deprecated::
    Use ExpectationTestGenerator instead. This class generates tests based
    on Oracle output, which can lead to tests that "adapt" to Oracle errors
    instead of verifying task requirements.
"""

import json
import re
import warnings
from typing import Any

from ..config import FileOperation, FileOperationType, ProcessedTrajectory, Trajectory
from ..execution.trajectory_recorder import TrajectoryRecorder


# Test file template
TEST_TEMPLATE = '''"""Auto-generated tests for Harbor task outputs"""

import json
import os

import pytest


class TestOutputs:
    """Test class for validating task outputs"""

{test_methods}
'''

# Individual test method templates
FILE_EXISTS_TEST = '''    def test_{test_name}_exists(self):
        """Check that {file_desc} was created."""
        assert os.path.exists("{file_path}"), "Output file not found: {file_path}"
'''

JSON_FORMAT_TEST = '''    def test_{test_name}_json_valid(self):
        """Validate {file_desc} is valid JSON."""
        with open("{file_path}") as f:
            data = json.load(f)
        assert data is not None, "JSON file is empty"
'''

JSON_KEYS_TEST = '''    def test_{test_name}_json_keys(self):
        """Validate {file_desc} contains required keys."""
        with open("{file_path}") as f:
            data = json.load(f)
        required_keys = {required_keys}
        for key in required_keys:
            assert key in data, f"Missing required key: {{key}}"
'''

JSON_VALUES_TEST = '''    def test_{test_name}_json_values(self):
        """Validate {file_desc} values."""
        with open("{file_path}") as f:
            data = json.load(f)
{value_assertions}
'''

CSV_HEADER_TEST = '''    def test_{test_name}_csv_header(self):
        """Validate {file_desc} has correct header."""
        with open("{file_path}") as f:
            header = f.readline().strip().split(",")
        expected_columns = {expected_columns}
        for col in expected_columns:
            assert col in header, f"Missing column: {{col}}"
'''

CSV_DATA_TEST = '''    def test_{test_name}_csv_has_data(self):
        """Validate {file_desc} contains data rows."""
        with open("{file_path}") as f:
            lines = f.readlines()
        assert len(lines) > 1, "CSV file has no data rows"
'''

TEXT_NOT_EMPTY_TEST = '''    def test_{test_name}_not_empty(self):
        """Validate {file_desc} is not empty."""
        with open("{file_path}") as f:
            content = f.read().strip()
        assert len(content) > 0, "File is empty"
'''

TEXT_CONTAINS_TEST = '''    def test_{test_name}_contains_expected(self):
        """Validate {file_desc} contains expected content."""
        with open("{file_path}") as f:
            content = f.read().lower()
        expected_patterns = {expected_patterns}
        for pattern in expected_patterns:
            assert pattern.lower() in content, f"Missing expected content: {{pattern}}"
'''

MARKDOWN_HEADERS_TEST = '''    def test_{test_name}_md_structure(self):
        """Validate {file_desc} has markdown structure."""
        with open("{file_path}") as f:
            content = f.read()
        # Check for at least one header
        assert "#" in content, "Markdown file has no headers"
'''


class TestGenerator:
    """Generates pytest tests from trajectory outputs.

    .. deprecated::
        Use ExpectationTestGenerator instead. This class generates tests based
        on Oracle output, which can lead to tests that "adapt" to Oracle errors
        instead of verifying task requirements.
    """

    def __init__(self):
        warnings.warn(
            "TestGenerator is deprecated. Use ExpectationTestGenerator instead. "
            "This class generates tests based on Oracle output, which may not "
            "correctly verify task completion.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.recorder = TrajectoryRecorder()

    def generate(
        self,
        trajectory: Trajectory,
        output_files: list[str] | None = None,
    ) -> str:
        """
        Generate test_outputs.py content based on trajectory.

        Args:
            trajectory: Execution trajectory
            output_files: Override output files list

        Returns:
            Complete test file content
        """
        files = output_files or trajectory.output_files
        contents = self.recorder.get_output_file_contents(trajectory)

        test_methods = []

        # If no explicit output files, try to get them from write operations
        if not files:
            files = list(contents.keys())

        for file_path in files:
            content = contents.get(file_path)
            ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""

            # Skip directory paths (no extension, typically /root/output)
            if not ext and file_path.endswith("/output"):
                continue

            # Generate tests based on file type
            tests = self._generate_file_tests(file_path, ext, content)
            test_methods.extend(tests)

        if not test_methods:
            # Generate at least a basic existence test
            test_methods.append(self._generate_basic_test())

        return TEST_TEMPLATE.format(test_methods="\n".join(test_methods))

    def generate_from_processed(self, processed: ProcessedTrajectory) -> str:
        """Generate tests from a ProcessedTrajectory."""
        return self.generate(
            trajectory=processed.trajectory,
            output_files=processed.created_files or processed.trajectory.output_files,
        )

    def _generate_file_tests(
        self,
        file_path: str,
        extension: str,
        content: str | None,
    ) -> list[str]:
        """Generate appropriate tests for a file based on its type"""
        tests = []
        test_name = self._make_test_name(file_path)
        file_desc = file_path.split("/")[-1]

        # Always test file existence
        tests.append(FILE_EXISTS_TEST.format(
            test_name=test_name,
            file_desc=file_desc,
            file_path=file_path,
        ))

        if extension == "json":
            tests.extend(self._generate_json_tests(file_path, test_name, file_desc, content))
        elif extension == "csv":
            tests.extend(self._generate_csv_tests(file_path, test_name, file_desc, content))
        elif extension in ("md", "markdown"):
            tests.extend(self._generate_markdown_tests(file_path, test_name, file_desc, content))
        elif extension in ("txt", "text"):
            tests.extend(self._generate_text_tests(file_path, test_name, file_desc, content))
        else:
            # Generic non-empty test
            tests.append(TEXT_NOT_EMPTY_TEST.format(
                test_name=test_name,
                file_desc=file_desc,
                file_path=file_path,
            ))

        return tests

    def _generate_json_tests(
        self,
        file_path: str,
        test_name: str,
        file_desc: str,
        content: str | None,
    ) -> list[str]:
        """Generate tests for JSON files"""
        tests = []

        # Valid JSON test
        tests.append(JSON_FORMAT_TEST.format(
            test_name=test_name,
            file_desc=file_desc,
            file_path=file_path,
        ))

        # If we have content, generate key-based tests
        if content:
            try:
                data = json.loads(content)
                if isinstance(data, dict) and data:
                    keys = list(data.keys())
                    tests.append(JSON_KEYS_TEST.format(
                        test_name=test_name,
                        file_desc=file_desc,
                        file_path=file_path,
                        required_keys=repr(keys),
                    ))

                    # Generate value type assertions for simple values
                    value_assertions = []
                    for key, value in data.items():
                        if isinstance(value, (int, float)):
                            value_assertions.append(
                                f'        assert isinstance(data["{key}"], (int, float)), "{key} should be numeric"'
                            )
                        elif isinstance(value, str):
                            value_assertions.append(
                                f'        assert isinstance(data["{key}"], str), "{key} should be a string"'
                            )
                        elif isinstance(value, list):
                            value_assertions.append(
                                f'        assert isinstance(data["{key}"], list), "{key} should be a list"'
                            )
                        elif isinstance(value, dict):
                            value_assertions.append(
                                f'        assert isinstance(data["{key}"], dict), "{key} should be a dict"'
                            )

                    if value_assertions:
                        tests.append(JSON_VALUES_TEST.format(
                            test_name=test_name,
                            file_desc=file_desc,
                            file_path=file_path,
                            value_assertions="\n".join(value_assertions),
                        ))
            except json.JSONDecodeError:
                pass

        return tests

    def _generate_csv_tests(
        self,
        file_path: str,
        test_name: str,
        file_desc: str,
        content: str | None,
    ) -> list[str]:
        """Generate tests for CSV files"""
        tests = []

        # Has data test
        tests.append(CSV_DATA_TEST.format(
            test_name=test_name,
            file_desc=file_desc,
            file_path=file_path,
        ))

        # Header test if we have content
        if content:
            lines = content.strip().split("\n")
            if lines:
                header = [col.strip() for col in lines[0].split(",")]
                if header:
                    tests.append(CSV_HEADER_TEST.format(
                        test_name=test_name,
                        file_desc=file_desc,
                        file_path=file_path,
                        expected_columns=repr(header),
                    ))

        return tests

    def _generate_markdown_tests(
        self,
        file_path: str,
        test_name: str,
        file_desc: str,
        content: str | None,
    ) -> list[str]:
        """Generate tests for Markdown files"""
        tests = []

        # Not empty test
        tests.append(TEXT_NOT_EMPTY_TEST.format(
            test_name=test_name,
            file_desc=file_desc,
            file_path=file_path,
        ))

        # Structure test
        tests.append(MARKDOWN_HEADERS_TEST.format(
            test_name=test_name,
            file_desc=file_desc,
            file_path=file_path,
        ))

        return tests

    def _generate_text_tests(
        self,
        file_path: str,
        test_name: str,
        file_desc: str,
        content: str | None,
    ) -> list[str]:
        """Generate tests for text files"""
        tests = []

        # Not empty test
        tests.append(TEXT_NOT_EMPTY_TEST.format(
            test_name=test_name,
            file_desc=file_desc,
            file_path=file_path,
        ))

        return tests

    def _generate_basic_test(self) -> str:
        """Generate a basic placeholder test"""
        return '''    def test_outputs_exist(self):
        """Basic test - check that some output was produced."""
        import os
        output_dir = "/root/output"
        if os.path.isdir(output_dir):
            files = os.listdir(output_dir)
            assert len(files) > 0, "No output files produced"
        elif os.path.isfile(output_dir):
            # output_dir is actually a file
            with open(output_dir) as f:
                content = f.read()
            assert len(content) > 0, "Output file is empty"
        else:
            pytest.fail("Output not found at /root/output")
'''

    def _make_test_name(self, file_path: str) -> str:
        """Convert file path to valid test function name"""
        filename = file_path.split("/")[-1]
        # Remove extension
        if "." in filename:
            filename = filename.rsplit(".", 1)[0]
        # Replace invalid characters
        name = re.sub(r'[^a-zA-Z0-9_]', '_', filename)
        # Ensure starts with letter
        if name and name[0].isdigit():
            name = "file_" + name
        return name or "output"

    def generate_test_sh(self) -> str:
        """Generate the test.sh runner script"""
        return '''#!/bin/bash
pip3 install --break-system-packages pytest==8.4.1 pytest-json-ctrf==0.3.5
mkdir -p /logs/verifier
pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA -v
if [ $? -eq 0 ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
exit 0
'''
