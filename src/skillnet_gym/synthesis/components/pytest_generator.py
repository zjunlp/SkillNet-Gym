"""Pytest generator for skillsbench-style tests.

This module generates pytest tests that verify task outputs using
hardcoded expected values and structural checks.
"""

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..execution.claude_executor import ClaudeExecutor

from ..prompts.pytest_prompts import (
    PYTEST_GENERATION_PROMPT,
    PYTEST_GENERATION_SYSTEM,
    PYTEST_RETRY_PROMPT,
    format_files_info,
)
from ..utils.llm_client import LLMClient


class PytestGenerator:
    """Generates skillsbench-style pytest tests from oracle outputs.

    This generator creates tests that:
    1. Verify file existence and format
    2. Use hardcoded expected values extracted from actual outputs
    3. Include value, content, and structural verification
    """

    def __init__(
        self,
        executor: "ClaudeExecutor | None" = None,
        llm_client: LLMClient | None = None,
    ):
        """Initialize the generator.

        Args:
            executor: Claude Code executor for test generation
            llm_client: LLM client for fallback generation
        """
        self.executor = executor
        self.llm = llm_client

    def generate(
        self,
        task_instruction: str,
        trajectory_summary: str,
        final_files: list[str | Path],
        working_dir: str,
        test_file_path: str,
    ) -> str:
        """Generate pytest tests from final output files.

        Args:
            task_instruction: The original task instruction
            trajectory_summary: Summary of the oracle trajectory
            final_files: List of final output file paths
            working_dir: Working directory (where files are located)
            test_file_path: Path to save the test file

        Returns:
            Generated test code string
        """
        # Format files info
        files_str = [str(f) for f in final_files]
        final_files_info = format_files_info(files_str)

        # Build prompt
        prompt = PYTEST_GENERATION_PROMPT.format(
            task_instruction=task_instruction,
            trajectory_summary=trajectory_summary,
            final_files_info=final_files_info,
            test_file_path=test_file_path,
        )

        full_prompt = f"{PYTEST_GENERATION_SYSTEM}\n\n{prompt}"

        # Use executor if available
        if self.executor:
            return self._generate_with_executor(
                prompt=full_prompt,
                working_dir=working_dir,
                test_file_path=test_file_path,
            )

        # Fallback to LLM
        if self.llm:
            return self._generate_with_llm(full_prompt, test_file_path)

        raise RuntimeError("No executor or LLM client available for pytest generation")

    def regenerate_with_feedback(
        self,
        task_instruction: str,
        trajectory_summary: str,
        final_files: list[str | Path],
        previous_test: str,
        failure_info: str,
        working_dir: str,
        test_file_path: str,
    ) -> str:
        """Regenerate tests after failure.

        Args:
            task_instruction: The original task instruction
            trajectory_summary: Summary of the oracle trajectory
            final_files: List of final output file paths
            previous_test: Previously generated test code
            failure_info: Summary of test failures
            working_dir: Working directory
            test_file_path: Path to save the test file

        Returns:
            Regenerated test code string
        """
        # Format files info
        files_str = [str(f) for f in final_files]
        final_files_info = format_files_info(files_str)

        # Build regeneration prompt
        prompt = PYTEST_RETRY_PROMPT.format(
            task_instruction=task_instruction,
            trajectory_summary=trajectory_summary,
            final_files_info=final_files_info,
            previous_test=previous_test,
            failure_info=failure_info,
            test_file_path=test_file_path,
        )

        full_prompt = f"{PYTEST_GENERATION_SYSTEM}\n\n{prompt}"

        # Use executor if available
        if self.executor:
            return self._generate_with_executor(
                prompt=full_prompt,
                working_dir=working_dir,
                test_file_path=test_file_path,
            )

        # Fallback to LLM
        if self.llm:
            return self._generate_with_llm(full_prompt, test_file_path)

        raise RuntimeError("No executor or LLM client available for pytest regeneration")

    def _generate_with_executor(
        self,
        prompt: str,
        working_dir: str,
        test_file_path: str,
    ) -> str:
        """Generate tests using Claude Code executor.

        Args:
            prompt: The generation prompt
            working_dir: Working directory
            test_file_path: Path to save test file

        Returns:
            Generated test code
        """
        result = self.executor.execute(
            prompt=prompt,
            working_dir=working_dir,
        )

        # Check if test file was created
        test_path = Path(test_file_path)
        if test_path.exists():
            return test_path.read_text(encoding="utf-8")

        # Try to extract code from output
        if result.output:
            code = self._extract_code_from_output(result.output)
            if code:
                # Write to file
                test_path.write_text(code, encoding="utf-8")
                return code

        raise RuntimeError(f"Failed to generate pytest: {result.error or 'No output'}")

    def _generate_with_llm(self, prompt: str, test_file_path: str) -> str:
        """Generate tests using LLM.

        Args:
            prompt: The generation prompt
            test_file_path: Path to save test file

        Returns:
            Generated test code
        """
        response = self.llm.generate(
            system_prompt=PYTEST_GENERATION_SYSTEM,
            user_prompt=prompt,
            temperature=0.3,
        )

        # Extract code from response
        code = self._extract_code_from_output(response)
        if code:
            # Write to file
            Path(test_file_path).write_text(code, encoding="utf-8")
            return code

        # Return raw response if no code block found
        Path(test_file_path).write_text(response, encoding="utf-8")
        return response

    def _extract_code_from_output(self, output: str) -> str | None:
        """Extract Python code from model output.

        Args:
            output: Raw model output

        Returns:
            Extracted code or None
        """
        # Try to find code block
        patterns = [
            r'```python\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
        ]

        for pattern in patterns:
            match = re.search(pattern, output, re.DOTALL)
            if match:
                code = match.group(1)
                # Validate it looks like Python test code
                if "def test_" in code or "class Test" in code:
                    return code

        # Check if entire output is code
        if "def test_" in output or "class Test" in output:
            # Remove any non-code prefix
            lines = output.split("\n")
            code_start = 0
            for i, line in enumerate(lines):
                if line.startswith("import ") or line.startswith("from ") or line.startswith('"""') or line.startswith("class "):
                    code_start = i
                    break
            return "\n".join(lines[code_start:])

        return None

    def validate_test_syntax(self, code: str) -> tuple[bool, str]:
        """Validate that generated test code has valid Python syntax.

        Args:
            code: The test code to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            compile(code, "<string>", "exec")
            return True, ""
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
