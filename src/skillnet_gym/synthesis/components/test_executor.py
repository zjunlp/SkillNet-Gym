"""Test executor for running pytest tests and collecting results.

This module handles execution of expectation tests generated in Phase A
and collects structured results for verification.
"""

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TestFailure:
    """Represents a single test failure."""
    test_name: str
    error_message: str
    file_path: str = ""
    line_number: int = 0
    traceback: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_name": self.test_name,
            "error_message": self.error_message,
            "file_path": self.file_path,
            "line_number": self.line_number,
        }


@dataclass
class TestExecutionResult:
    """Result of test execution."""
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    total: int = 0
    failures: list[TestFailure] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    duration_seconds: float = 0.0

    @property
    def all_passed(self) -> bool:
        """Check if all tests passed.

        pytest return codes:
        - 0: all tests passed
        - 1: some tests failed
        - 2: test execution interrupted
        - 3: internal error
        - 4: command line usage error
        - 5: no tests collected
        """
        # Must have successful return code
        if self.return_code != 0:
            return False
        # Must have at least one passing test
        if self.passed == 0:
            return False
        # No failures or errors
        return self.failed == 0 and self.errors == 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total == 0:
            return 0.0
        return self.passed / self.total

    def get_failure_summary(self) -> str:
        """Get a summary of failures for feedback."""
        # Check for no tests collected (total=0, return_code=0)
        if self.total == 0 and self.return_code == 0:
            if self.stdout:
                return f"No tests collected (test file may be empty or have no test functions). Output:\n{self.stdout[:500]}"
            return "No tests collected (test file may be empty or have no test functions)"

        # Check for errors first (collection/import errors)
        if self.errors > 0 and not self.failures:
            # Provide stderr if available for debugging
            if self.stderr:
                return f"{self.errors} test error(s) (collection/import). Details:\n{self.stderr[:500]}"
            if self.stdout:
                return f"{self.errors} test error(s) (collection/import). Output:\n{self.stdout[:500]}"
            return f"{self.errors} test error(s) occurred (collection/import errors, return_code={self.return_code})"

        # Check for failed tests
        if self.failed > 0 and not self.failures:
            return f"{self.failed} test(s) failed (no detailed failure info available)"

        # No failures and no errors - truly passed
        if not self.failures and self.passed > 0:
            return "All tests passed."

        # No failures but also no passed tests (edge case)
        if not self.failures:
            return f"No test results (passed={self.passed}, failed={self.failed}, errors={self.errors})"

        lines = [f"Failed {len(self.failures)} test(s):"]
        for f in self.failures:
            lines.append(f"  - {f.test_name}: {f.error_message}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
            "total": self.total,
            "all_passed": self.all_passed,
            "success_rate": self.success_rate,
            "failures": [f.to_dict() for f in self.failures],
            "return_code": self.return_code,
            "duration_seconds": self.duration_seconds,
        }


class TestExecutor:
    """Executes pytest tests and collects structured results."""

    def __init__(self, python_path: str = "python3", conda_env: str | None = None):
        """
        Initialize the executor.

        Args:
            python_path: Path to Python interpreter
            conda_env: Conda environment name (e.g., "skillgym_bioinformatics_00").
                       If provided, tests run via `conda run -n <env>`.
        """
        self.python_path = python_path
        self.conda_env = conda_env

    def run_tests(
        self,
        test_file: str,
        working_dir: str | None = None,
        timeout: int = 120,
        verbose: bool = True,
    ) -> TestExecutionResult:
        """
        Execute pytest tests and collect results.

        Args:
            test_file: Path to test file
            working_dir: Working directory for test execution
            timeout: Timeout in seconds
            verbose: Enable verbose output

        Returns:
            TestExecutionResult with detailed results
        """
        test_path = Path(test_file)
        if not test_path.exists():
            return TestExecutionResult(
                errors=1,
                total=1,
                failures=[TestFailure(
                    test_name="test_file_exists",
                    error_message=f"Test file not found: {test_file}",
                )],
                return_code=1,
            )

        # Build pytest command
        pytest_args = [
            self.python_path, "-m", "pytest",
            str(test_path),
            "-v",
            "--tb=short",
            "-rA",  # Show all test results
        ]

        if verbose:
            pytest_args.append("-s")

        # Wrap with conda run if conda_env is specified
        if self.conda_env and self.conda_env != "base":
            cmd = [
                "conda", "run", "-n", self.conda_env, "--no-capture-output",
            ] + pytest_args
        else:
            cmd = pytest_args

        # Execute pytest
        try:
            result = subprocess.run(
                cmd,
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            # Parse results
            execution_result = self._parse_pytest_output(
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
            )

            return execution_result

        except subprocess.TimeoutExpired:
            return TestExecutionResult(
                errors=1,
                total=1,
                failures=[TestFailure(
                    test_name="execution",
                    error_message=f"Test execution timed out after {timeout} seconds",
                )],
                return_code=-1,
            )

        except Exception as e:
            return TestExecutionResult(
                errors=1,
                total=1,
                failures=[TestFailure(
                    test_name="execution",
                    error_message=f"Test execution failed: {str(e)}",
                )],
                return_code=-1,
            )

    def _parse_pytest_output(
        self,
        stdout: str,
        stderr: str,
        return_code: int,
    ) -> TestExecutionResult:
        """
        Parse pytest output to extract test results.

        Args:
            stdout: Standard output from pytest
            stderr: Standard error from pytest
            return_code: Process return code

        Returns:
            TestExecutionResult with parsed information
        """
        result = TestExecutionResult(
            stdout=stdout,
            stderr=stderr,
            return_code=return_code,
        )

        combined_output = stdout + stderr

        # Parse summary line (e.g., "1 passed, 2 failed in 0.12s")
        summary_match = re.search(
            r'=+ (.*?) in ([\d.]+)s =+',
            combined_output
        )

        if summary_match:
            summary = summary_match.group(1)
            duration = float(summary_match.group(2))
            result.duration_seconds = duration

            # Parse counts
            passed_match = re.search(r'(\d+) passed', summary)
            failed_match = re.search(r'(\d+) failed', summary)
            error_match = re.search(r'(\d+) error', summary)
            skipped_match = re.search(r'(\d+) skipped', summary)

            if passed_match:
                result.passed = int(passed_match.group(1))
            if failed_match:
                result.failed = int(failed_match.group(1))
            if error_match:
                result.errors = int(error_match.group(1))
            if skipped_match:
                result.skipped = int(skipped_match.group(1))

            result.total = result.passed + result.failed + result.errors + result.skipped

        # Parse individual failures
        result.failures = self._parse_failures(combined_output)

        # If we didn't find summary but have failures, estimate counts
        if result.total == 0 and result.failures:
            result.failed = len(result.failures)
            result.total = result.failed

        return result

    def _parse_failures(self, output: str) -> list[TestFailure]:
        """
        Parse test failures from pytest output.

        Args:
            output: Combined pytest output

        Returns:
            List of TestFailure objects
        """
        failures = []

        # Pattern 1: FAILED test_file.py::TestClass::test_name - AssertionError: message
        failed_pattern = re.compile(
            r'FAILED\s+([\w./]+)::(\w+)::(\w+)\s*[-–]\s*(.+?)(?=\n|$)',
            re.MULTILINE
        )

        for match in failed_pattern.finditer(output):
            file_path = match.group(1)
            class_name = match.group(2)
            test_name = match.group(3)
            error_msg = match.group(4).strip()

            failures.append(TestFailure(
                test_name=f"{class_name}::{test_name}",
                error_message=error_msg,
                file_path=file_path,
            ))

        # Pattern 2: AssertionError lines in output
        if not failures:
            # Look for assertion errors in traceback
            assertion_pattern = re.compile(
                r'(test_\w+).*?AssertionError:\s*(.+?)(?=\n|$)',
                re.DOTALL
            )

            for match in assertion_pattern.finditer(output):
                test_name = match.group(1)
                error_msg = match.group(2).strip()

                # Avoid duplicates
                if not any(f.test_name.endswith(test_name) for f in failures):
                    failures.append(TestFailure(
                        test_name=test_name,
                        error_message=error_msg,
                    ))

        # Pattern 3: Look for error sections
        error_section_pattern = re.compile(
            r'_+ (test_\w+) _+\s*(.*?)(?=_+|={3,}|\Z)',
            re.DOTALL
        )

        for match in error_section_pattern.finditer(output):
            test_name = match.group(1)
            section_content = match.group(2)

            # Extract error message
            error_match = re.search(
                r'(?:AssertionError|Error|Exception):\s*(.+?)(?=\n\n|\Z)',
                section_content,
                re.DOTALL
            )

            if error_match:
                error_msg = error_match.group(1).strip()
                # Clean up multiline
                error_msg = ' '.join(error_msg.split())

                # Avoid duplicates
                if not any(f.test_name == test_name for f in failures):
                    failures.append(TestFailure(
                        test_name=test_name,
                        error_message=error_msg[:200],  # Truncate long messages
                    ))

        return failures

    def validate_test_file(self, test_file: str) -> tuple[bool, str]:
        """
        Validate that a test file is syntactically correct.

        Args:
            test_file: Path to test file

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                code = f.read()

            compile(code, test_file, 'exec')
            return True, ""

        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def run_quick_check(
        self,
        test_file: str,
        working_dir: str | None = None,
    ) -> bool:
        """
        Run a quick check to see if tests can be collected.

        Args:
            test_file: Path to test file
            working_dir: Working directory

        Returns:
            True if tests can be collected
        """
        pytest_args = [
            self.python_path, "-m", "pytest",
            test_file,
            "--collect-only",
            "-q",
        ]

        # Wrap with conda run if conda_env is specified
        if self.conda_env and self.conda_env != "base":
            cmd = [
                "conda", "run", "-n", self.conda_env, "--no-capture-output",
            ] + pytest_args
        else:
            cmd = pytest_args

        try:
            result = subprocess.run(
                cmd,
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0

        except Exception:
            return False
