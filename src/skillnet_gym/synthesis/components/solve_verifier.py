"""Verifier for solve.sh scripts - ensures they pass tests in isolated environments"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ..utils.file_utils import (
    cleanup_directory,
    copy_directory,
    copy_file,
    ensure_directory,
    write_file_content,
)
from .test_executor import TestExecutionResult, TestExecutor


class SolveShVerifier:
    """Verifies solve.sh scripts by running them in isolated environments"""

    def __init__(
        self,
        test_executor: TestExecutor | None = None,
        python_path: str = "python3",
        conda_env: str = "base",
    ):
        """
        Initialize the verifier.

        Args:
            test_executor: TestExecutor instance for running tests
            python_path: Path to Python interpreter
            conda_env: Conda environment name for execution
        """
        self.test_executor = test_executor or TestExecutor(python_path=python_path)
        self.python_path = python_path
        self.conda_env = conda_env

    def setup_verification_env(
        self,
        input_files: list[str],
        skills_dir: str | None = None,
        base_dir: str | None = None,
    ) -> str:
        """
        Create an isolated verification environment.

        Args:
            input_files: List of input file paths to copy
            skills_dir: Path to skills directory (optional)
            base_dir: Base directory for temp env (uses system temp if None)

        Returns:
            Path to the verification environment directory
        """
        # Create temp directory
        if base_dir:
            ensure_directory(base_dir)
            env_dir = tempfile.mkdtemp(prefix="solve_verify_", dir=base_dir)
        else:
            env_dir = tempfile.mkdtemp(prefix="solve_verify_")

        env_path = Path(env_dir)

        # Copy input files
        for input_file in input_files:
            src = Path(input_file)
            if src.exists():
                # Copy to root of env_dir (same as container /root/)
                copy_file(src, env_path / src.name)

        # Copy skills to .claude/skills/
        if skills_dir:
            skills_path = Path(skills_dir)
            if skills_path.exists():
                claude_skills = env_path / ".claude" / "skills"
                copy_directory(
                    skills_path,
                    claude_skills,
                    ignore_patterns=["__pycache__", "*.pyc"],
                )

        return env_dir

    def execute_solve_sh(
        self,
        solve_sh_content: str,
        working_dir: str,
        timeout: int = 120,
        conda_env: str | None = None,
    ) -> tuple[bool, str, str]:
        """
        Execute solve.sh in the working directory.

        Args:
            solve_sh_content: Content of solve.sh script
            working_dir: Working directory for execution
            timeout: Timeout in seconds
            conda_env: Conda environment to use (overrides self.conda_env if provided)

        Returns:
            Tuple of (success, stdout, stderr)
        """
        work_path = Path(working_dir)
        env_name = conda_env or self.conda_env

        # Write solve.sh
        solve_sh_path = work_path / "solve.sh"
        write_file_content(solve_sh_path, solve_sh_content)

        # Make executable
        os.chmod(solve_sh_path, 0o755)

        # Build command - use conda run for non-base environments
        if env_name and env_name != "base":
            cmd = ["conda", "run", "-n", env_name, "--no-capture-output", "bash", str(solve_sh_path)]
        else:
            cmd = ["bash", str(solve_sh_path)]

        # Execute
        try:
            result = subprocess.run(
                cmd,
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "HOME": working_dir},
            )

            success = result.returncode == 0
            return success, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            return False, "", f"solve.sh execution timed out after {timeout} seconds"
        except Exception as e:
            return False, "", f"solve.sh execution failed: {str(e)}"

    def verify(
        self,
        solve_sh_content: str,
        test_file: str,
        input_files: list[str],
        skills_dir: str | None = None,
        timeout: int = 120,
        base_dir: str | None = None,
        conda_env: str | None = None,
    ) -> tuple[bool, TestExecutionResult]:
        """
        Verify that solve.sh passes the tests.

        This method:
        1. Creates an isolated verification environment
        2. Copies input files and skills
        3. Executes solve.sh
        4. Runs the test file against the results
        5. Cleans up the environment

        Args:
            solve_sh_content: Content of solve.sh script
            test_file: Path to the test file
            input_files: List of input file paths
            skills_dir: Path to skills directory
            timeout: Timeout for solve.sh execution
            base_dir: Base directory for temp env
            conda_env: Conda environment to use (overrides self.conda_env if provided)

        Returns:
            Tuple of (passed: bool, test_result: TestExecutionResult)
        """
        env_dir = None

        try:
            # 1. Setup verification environment
            env_dir = self.setup_verification_env(
                input_files=input_files,
                skills_dir=skills_dir,
                base_dir=base_dir,
            )
            env_path = Path(env_dir)

            # 2. Copy test file to environment
            test_src = Path(test_file)
            if not test_src.exists():
                return False, TestExecutionResult(
                    errors=1,
                    total=1,
                    failures=[],
                    return_code=-1,
                    stdout="",
                    stderr=f"Test file not found: {test_file}",
                )

            test_dest = env_path / test_src.name
            copy_file(test_src, test_dest)

            # 3. Execute solve.sh (with conda environment support)
            solve_success, solve_stdout, solve_stderr = self.execute_solve_sh(
                solve_sh_content=solve_sh_content,
                working_dir=env_dir,
                timeout=timeout,
                conda_env=conda_env,
            )

            if not solve_success:
                # solve.sh failed - create failure result
                from .test_executor import TestFailure
                return False, TestExecutionResult(
                    errors=1,
                    total=1,
                    failures=[TestFailure(
                        test_name="solve_sh_execution",
                        error_message=f"solve.sh failed: {solve_stderr}",
                    )],
                    return_code=-1,
                    stdout=solve_stdout,
                    stderr=solve_stderr,
                )

            # 4. Run tests
            test_result = self.test_executor.run_tests(
                test_file=str(test_dest),
                working_dir=env_dir,
                timeout=timeout,
                verbose=True,
            )

            return test_result.all_passed, test_result

        finally:
            # 5. Cleanup
            if env_dir:
                self.cleanup_verification_env(env_dir)

    def cleanup_verification_env(self, env_dir: str) -> bool:
        """
        Clean up the verification environment.

        Args:
            env_dir: Path to the environment directory

        Returns:
            True if cleanup succeeded
        """
        return cleanup_directory(env_dir)

    def verify_in_clean_workspace(
        self,
        solve_sh_content: str,
        test_file_content: str,
        input_files: list[str],
        skills_dir: str | None = None,
        timeout: int = 120,
        conda_env: str | None = None,
        cleanup: bool = False,
    ) -> tuple[bool, TestExecutionResult, str]:
        """
        Verify solve.sh in a clean workspace with test content.

        This method creates an isolated verification environment, writes the
        test file content directly (instead of copying from a path), and
        returns the workspace path for inspection.

        Args:
            solve_sh_content: Content of solve.sh script
            test_file_content: Content of the pytest test file
            input_files: List of input file paths to copy
            skills_dir: Path to skills directory
            timeout: Timeout for solve.sh execution
            conda_env: Conda environment to use
            cleanup: Whether to cleanup workspace after verification

        Returns:
            Tuple of (passed: bool, test_result: TestExecutionResult, workspace_path: str)
        """
        env_dir = None

        try:
            # 1. Setup verification environment
            env_dir = self.setup_verification_env(
                input_files=input_files,
                skills_dir=skills_dir,
            )
            env_path = Path(env_dir)

            # 2. Write test file content directly
            test_dest = env_path / "test_outputs.py"
            test_dest.write_text(test_file_content, encoding="utf-8")

            # 3. Execute solve.sh (with conda environment support)
            solve_success, solve_stdout, solve_stderr = self.execute_solve_sh(
                solve_sh_content=solve_sh_content,
                working_dir=env_dir,
                timeout=timeout,
                conda_env=conda_env,
            )

            if not solve_success:
                # solve.sh failed - create failure result
                from .test_executor import TestFailure
                return False, TestExecutionResult(
                    errors=1,
                    total=1,
                    failures=[TestFailure(
                        test_name="solve_sh_execution",
                        error_message=f"solve.sh failed: {solve_stderr}",
                    )],
                    return_code=-1,
                    stdout=solve_stdout,
                    stderr=solve_stderr,
                ), env_dir

            # 4. Run tests
            test_result = self.test_executor.run_tests(
                test_file=str(test_dest),
                working_dir=env_dir,
                timeout=timeout,
                verbose=True,
            )

            return test_result.all_passed, test_result, env_dir

        finally:
            # 5. Cleanup if requested
            if cleanup and env_dir:
                self.cleanup_verification_env(env_dir)

    def verify_with_details(
        self,
        solve_sh_content: str,
        test_file: str,
        input_files: list[str],
        skills_dir: str | None = None,
        timeout: int = 120,
        keep_env: bool = False,
        conda_env: str | None = None,
    ) -> dict[str, Any]:
        """
        Verify solve.sh with detailed results for debugging.

        Similar to verify() but returns more information and optionally
        keeps the environment for inspection.

        Args:
            solve_sh_content: Content of solve.sh script
            test_file: Path to the test file
            input_files: List of input file paths
            skills_dir: Path to skills directory
            timeout: Timeout for solve.sh execution
            keep_env: If True, don't cleanup the environment
            conda_env: Conda environment to use (overrides self.conda_env if provided)

        Returns:
            Dictionary with detailed verification results:
            {
                "passed": bool,
                "env_dir": str (path, only if keep_env=True),
                "solve_sh_success": bool,
                "solve_sh_stdout": str,
                "solve_sh_stderr": str,
                "test_result": TestExecutionResult,
                "output_files": list[str],
            }
        """
        env_dir = None

        try:
            # Setup
            env_dir = self.setup_verification_env(
                input_files=input_files,
                skills_dir=skills_dir,
            )
            env_path = Path(env_dir)

            # Copy test file
            test_src = Path(test_file)
            test_dest = env_path / test_src.name
            if test_src.exists():
                copy_file(test_src, test_dest)

            # Execute solve.sh (with conda environment support)
            solve_success, solve_stdout, solve_stderr = self.execute_solve_sh(
                solve_sh_content=solve_sh_content,
                working_dir=env_dir,
                timeout=timeout,
                conda_env=conda_env,
            )

            # List output files
            output_files = []
            for f in env_path.rglob("*"):
                if f.is_file() and f.name not in ("solve.sh", test_src.name):
                    output_files.append(str(f.relative_to(env_path)))

            # Run tests if solve.sh succeeded
            if solve_success and test_src.exists():
                test_result = self.test_executor.run_tests(
                    test_file=str(test_dest),
                    working_dir=env_dir,
                    timeout=timeout,
                    verbose=True,
                )
            else:
                from .test_executor import TestFailure
                test_result = TestExecutionResult(
                    errors=1 if not solve_success else 0,
                    total=1,
                    failures=[TestFailure(
                        test_name="solve_sh_execution" if not solve_success else "test_setup",
                        error_message=solve_stderr if not solve_success else "Test file not found",
                    )],
                    stdout=solve_stdout,
                    stderr=solve_stderr,
                )

            result = {
                "passed": test_result.all_passed and solve_success,
                "solve_sh_success": solve_success,
                "solve_sh_stdout": solve_stdout,
                "solve_sh_stderr": solve_stderr,
                "test_result": test_result,
                "output_files": output_files,
            }

            if keep_env:
                result["env_dir"] = env_dir
                env_dir = None  # Prevent cleanup

            return result

        finally:
            if env_dir and not keep_env:
                self.cleanup_verification_env(env_dir)
