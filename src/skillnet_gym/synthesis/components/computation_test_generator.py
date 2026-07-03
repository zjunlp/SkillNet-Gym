"""Computation-based test generator.

This module generates pytest tests that verify oracle outputs through
independent computation, rather than hardcoded expected values.
"""

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..execution.claude_executor import ClaudeExecutor

from ..config import Trajectory
from ..prompts.computation_test_prompts import (
    COMPUTATION_TEST_PROMPT,
    COMPUTATION_TEST_REGENERATION_PROMPT,
    COMPUTATION_TEST_SYSTEM,
    STRUCTURAL_TEST_TEMPLATE,
    format_trajectory_for_prompt,
)
from ..utils.llm_client import LLMClient
from .test_executor import TestExecutionResult


class ComputationTestGenerator:
    """Generates computation-based verification tests from oracle trajectory.

    Instead of hardcoding expected values, this generator creates tests that:
    1. Read the oracle's output files
    2. Independently compute what the expected values should be
    3. Compare and assert that oracle output matches computation
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
        trajectory: Trajectory,
        task_instruction: str,
        input_files: list[str],
        output_files: list[str],
        working_dir: str,
        skills_dir: str,
        test_file_path: str | None = None,
    ) -> str:
        """Generate computation-based tests from oracle trajectory.

        Args:
            trajectory: The oracle execution trajectory
            task_instruction: The original task instruction
            input_files: List of input file paths
            output_files: List of output file paths
            working_dir: Working directory
            skills_dir: Skills directory path
            test_file_path: Path to save test file (optional)

        Returns:
            Generated test code string
        """
        if test_file_path is None:
            test_file_path = str(Path(working_dir) / "computation_tests.py")

        # Format trajectory for prompt
        trajectory_steps = self._extract_trajectory_steps(trajectory)
        trajectory_str = format_trajectory_for_prompt(trajectory_steps)

        # Format file lists
        input_files_str = "\n".join(f"- {f}" for f in input_files)
        output_files_str = "\n".join(f"- {f}" for f in output_files)

        # Build prompt
        prompt = f"{COMPUTATION_TEST_SYSTEM}\n\n{COMPUTATION_TEST_PROMPT.format(task_instruction=task_instruction, trajectory=trajectory_str, input_files=input_files_str, output_files=output_files_str, test_file_path=test_file_path)}"

        # Use executor if available
        if self.executor:
            return self._generate_with_executor(
                prompt=prompt,
                working_dir=working_dir,
                skills_dir=skills_dir,
                test_file_path=test_file_path,
            )

        # Fallback to LLM
        if self.llm:
            return self._generate_with_llm(prompt)

        # Fallback to structural tests only
        return self._generate_structural_tests(output_files)

    def regenerate(
        self,
        trajectory: Trajectory,
        task_instruction: str,
        failure_summary: str,
        working_dir: str,
        skills_dir: str,
        test_file_path: str | None = None,
    ) -> str:
        """Regenerate tests after failure.

        Args:
            trajectory: The oracle execution trajectory
            task_instruction: The original task instruction
            failure_summary: Summary of test failures
            working_dir: Working directory
            skills_dir: Skills directory path
            test_file_path: Path to save test file

        Returns:
            Regenerated test code string
        """
        if test_file_path is None:
            test_file_path = str(Path(working_dir) / "computation_tests.py")

        # Format trajectory
        trajectory_steps = self._extract_trajectory_steps(trajectory)
        trajectory_str = format_trajectory_for_prompt(trajectory_steps)

        # Build regeneration prompt
        prompt = COMPUTATION_TEST_REGENERATION_PROMPT.format(
            failure_summary=failure_summary,
            task_instruction=task_instruction,
            trajectory=trajectory_str,
            test_file_path=test_file_path,
        )

        full_prompt = f"{COMPUTATION_TEST_SYSTEM}\n\n{prompt}"

        # Use executor if available
        if self.executor:
            return self._generate_with_executor(
                prompt=full_prompt,
                working_dir=working_dir,
                skills_dir=skills_dir,
                test_file_path=test_file_path,
            )

        # Fallback to LLM
        if self.llm:
            return self._generate_with_llm(full_prompt)

        raise RuntimeError("No executor or LLM client available for regeneration")

    def _extract_trajectory_steps(self, trajectory: Trajectory) -> list[dict]:
        """Extract trajectory steps as dictionaries for prompt.

        Args:
            trajectory: The trajectory to extract

        Returns:
            List of step dictionaries
        """
        steps = []
        for step in trajectory.steps:
            if step.action_type != "tool_use":
                continue

            step_data = {
                "tool": step.tool_name,
                "action": step.tool_input,
            }

            # Include response preview for context
            if step.tool_output:
                output_preview = step.tool_output[:500]
                if len(step.tool_output) > 500:
                    output_preview += "..."
                step_data["response"] = output_preview

            steps.append(step_data)

        return steps

    def _generate_with_executor(
        self,
        prompt: str,
        working_dir: str,
        skills_dir: str,
        test_file_path: str,
    ) -> str:
        """Generate tests using Claude Code executor.

        Args:
            prompt: The generation prompt
            working_dir: Working directory
            skills_dir: Skills directory
            test_file_path: Path to save test file

        Returns:
            Generated test code
        """
        result = self.executor.execute(
            prompt=prompt,
            working_dir=working_dir,
            skills_dir=skills_dir,
        )

        # Check if test file was created
        test_path = Path(test_file_path)
        if test_path.exists():
            return test_path.read_text(encoding="utf-8")

        # Try to extract code from output
        if result.output:
            code = self._extract_code_from_output(result.output)
            if code:
                return code

        raise RuntimeError(f"Failed to generate computation tests: {result.error or 'No output'}")

    def _generate_with_llm(self, prompt: str) -> str:
        """Generate tests using LLM.

        Args:
            prompt: The generation prompt

        Returns:
            Generated test code
        """
        response = self.llm.generate(
            system_prompt=COMPUTATION_TEST_SYSTEM,
            user_prompt=prompt,
            temperature=0.3,
        )

        # Extract code from response
        code = self._extract_code_from_output(response)
        if code:
            return code

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
                if line.startswith("import ") or line.startswith("from ") or line.startswith('"""'):
                    code_start = i
                    break
            return "\n".join(lines[code_start:])

        return None

    def _generate_structural_tests(self, output_files: list[str]) -> str:
        """Generate minimal structural tests as fallback.

        Args:
            output_files: List of output file paths

        Returns:
            Structural test code
        """
        file_existence_tests = []
        format_validity_tests = []
        structure_tests = []

        for i, file_path in enumerate(output_files):
            name = Path(file_path).stem.replace("-", "_").replace(".", "_")
            ext = Path(file_path).suffix.lower()

            # File existence test
            file_existence_tests.append(f'''
    def test_{name}_exists(self):
        """Verify {file_path} was created."""
        assert os.path.exists("{file_path}"), "Output file not found: {file_path}"
''')

            # Format validity test
            if ext == ".json":
                format_validity_tests.append(f'''
    def test_{name}_valid_json(self):
        """Verify {file_path} is valid JSON."""
        with open("{file_path}") as f:
            data = json.load(f)
        assert data is not None, "JSON file is empty or null"
''')
            elif ext == ".csv":
                format_validity_tests.append(f'''
    def test_{name}_has_data(self):
        """Verify {file_path} has data rows."""
        with open("{file_path}") as f:
            lines = f.readlines()
        assert len(lines) > 1, "CSV file has no data rows"
''')
            else:
                format_validity_tests.append(f'''
    def test_{name}_not_empty(self):
        """Verify {file_path} is not empty."""
        assert os.path.getsize("{file_path}") > 0, "File is empty"
''')

        return STRUCTURAL_TEST_TEMPLATE.format(
            file_existence_tests="\n".join(file_existence_tests),
            format_validity_tests="\n".join(format_validity_tests),
            structure_tests="\n".join(structure_tests),
        )

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
