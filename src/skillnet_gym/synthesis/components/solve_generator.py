"""Generator for solve.sh scripts from execution trajectories"""

import re
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..execution.claude_executor import ClaudeExecutor

from ..config import Trajectory, TrajectoryStep
from ..prompts.solve_sh_prompts import (
    CONSTANTS_REFERENCE_SECTION,
    GENERATE_SOLVE_SH_SYSTEM,
    GENERATE_SOLVE_SH_USER,
    GENERATE_SOLVE_WITH_EXECUTOR_SYSTEM,
    GENERATE_SOLVE_WITH_EXECUTOR_USER,
    REFINE_SOLVE_WITH_EXECUTOR_PROMPT,
    REGENERATE_SOLVE_SH_SYSTEM,
    REGENERATE_SOLVE_SH_USER,
)
from ..utils.llm_client import LLMClient
from .test_executor import TestFailure
from .trajectory_processor import TrajectoryProcessor


# solve.sh template
SOLVE_SH_TEMPLATE = '''#!/bin/bash
set -e

# Auto-generated solution based on execution trajectory
# This script recreates the outputs produced during synthesis

{script_content}
'''


class SolveShGenerator:
    """Generates solve.sh scripts from execution trajectories"""

    def __init__(self, llm_client: LLMClient | None = None):
        """
        Initialize the generator.

        Args:
            llm_client: LLM client for regeneration (optional)
        """
        self.llm_client = llm_client
        self.trajectory_processor = TrajectoryProcessor()

    def generate_from_trajectory(self, trajectory: Trajectory) -> str:
        """
        Generate solve.sh from trajectory.

        Enhanced version that handles:
        - Write: File creation via heredoc
        - Bash: Command execution
        - Edit: File editing via Python script
        - NotebookEdit: Jupyter notebook editing via Python script

        Args:
            trajectory: Execution trajectory

        Returns:
            solve.sh content string
        """
        script_lines = []

        for step in trajectory.steps:
            if step.action_type != "tool_use":
                continue

            handler = self._get_step_handler(step)
            if handler:
                lines = handler(step)
                if lines:
                    script_lines.extend(lines)
                    script_lines.append("")

        if not script_lines:
            script_lines.append("echo 'No automated solution available'")
            script_lines.append("echo 'Please implement the task manually'")

        solve_sh = SOLVE_SH_TEMPLATE.format(
            script_content="\n".join(script_lines),
        )

        # Sanitize to remove test file imports
        return self._sanitize_solve_sh(solve_sh)

    def generate_solve_sh(
        self,
        trajectory: Trajectory,
        task_instruction: str | None = None,
        expectation_tests_content: str | None = None,
    ) -> str:
        """
        Generate solve.sh with automatic fallback to LLM when test imports are detected.

        This method:
        1. First tries trajectory-based generation
        2. If test imports are found (and removed), falls back to LLM generation
           to create a self-contained script without external dependencies

        Args:
            trajectory: Execution trajectory
            task_instruction: Task instruction (required for LLM fallback)
            expectation_tests_content: Content of expectation_tests.py (for extracting constants)

        Returns:
            solve.sh content string
        """
        script_lines = []

        for step in trajectory.steps:
            if step.action_type != "tool_use":
                continue

            handler = self._get_step_handler(step)
            if handler:
                lines = handler(step)
                if lines:
                    script_lines.extend(lines)
                    script_lines.append("")

        if not script_lines:
            script_lines.append("echo 'No automated solution available'")
            script_lines.append("echo 'Please implement the task manually'")

        solve_sh = SOLVE_SH_TEMPLATE.format(
            script_content="\n".join(script_lines),
        )

        # Sanitize and check if test imports were removed
        sanitized, had_imports_removed = self._sanitize_solve_sh(solve_sh, return_flag=True)

        # If test imports were removed, the script likely uses undefined constants
        # Fall back to LLM generation for a self-contained script
        if had_imports_removed and self.llm_client and task_instruction:
            print(f"[SolveShGenerator] Test imports detected, using LLM to generate self-contained solve.sh...")

            # Build constants section if test content is available
            constants_section = ""
            if expectation_tests_content:
                constants_content = self._extract_constants_from_tests(expectation_tests_content)
                if constants_content:
                    constants_section = CONSTANTS_REFERENCE_SECTION.format(
                        constants_content=constants_content,
                    )

            # Generate trajectory summary
            trajectory_summary = self.trajectory_processor.summarize_for_llm(trajectory)

            # Format output files
            output_files = "\n".join(f"- {f}" for f in trajectory.output_files)

            # Build prompt with constants info
            user_prompt = GENERATE_SOLVE_SH_USER.format(
                task_instruction=task_instruction,
                trajectory_summary=trajectory_summary,
                output_files=output_files,
            )

            # Append constants section if available
            if constants_section:
                user_prompt += f"\n{constants_section}"

            # Call LLM
            response = self.llm_client.generate(
                system_prompt=GENERATE_SOLVE_SH_SYSTEM,
                user_prompt=user_prompt,
                temperature=0.3,
            )

            # Clean up response and sanitize
            solve_sh = self._clean_solve_sh_response(response)
            solve_sh = self._sanitize_solve_sh(solve_sh)
            return solve_sh

        return sanitized

    def _get_step_handler(self, step: TrajectoryStep):
        """Get the appropriate handler for a trajectory step"""
        handlers = {
            "Write": self._handle_write,
            "Bash": self._handle_bash,
            "Edit": self._handle_edit,
            "NotebookEdit": self._handle_notebook_edit,
        }
        return handlers.get(step.tool_name)

    def _handle_write(self, step: TrajectoryStep) -> list[str]:
        """Handle Write tool - create file via heredoc"""
        if not step.tool_input:
            return []

        file_path = step.tool_input.get("file_path", "")
        content = step.tool_input.get("content", "")

        if not file_path or not content:
            return []

        lines = [
            f"# Create {file_path}",
            f"mkdir -p $(dirname {file_path})",
            f"cat << 'HEREDOC_EOF' > {file_path}",
            content,
            "HEREDOC_EOF",
        ]
        return lines

    def _handle_bash(self, step: TrajectoryStep) -> list[str]:
        """Handle Bash tool - execute command"""
        if not step.tool_input:
            return []

        command = step.tool_input.get("command", "")
        if not command:
            return []

        # Skip certain commands that shouldn't be in solve.sh
        skip_patterns = [
            r"^cat\s+",      # Reading files
            r"^ls\s*",      # Listing directories
            r"^pwd\s*$",    # Current directory
            r"^cd\s+",      # Change directory (handled differently)
        ]
        for pattern in skip_patterns:
            if re.match(pattern, command.strip()):
                return []

        return [
            "# Execute command",
            command,
        ]

    def _handle_edit(self, step: TrajectoryStep) -> list[str]:
        """Handle Edit tool - file editing via Python script"""
        if not step.tool_input:
            return []

        file_path = step.tool_input.get("file_path", "")
        old_string = step.tool_input.get("old_string", "")
        new_string = step.tool_input.get("new_string", "")
        replace_all = step.tool_input.get("replace_all", False)

        if not file_path or old_string is None or new_string is None:
            return []

        # Use Python for robust string replacement
        # Escape special characters for Python string
        old_escaped = self._escape_python_string(old_string)
        new_escaped = self._escape_python_string(new_string)

        if replace_all:
            replace_method = "content.replace(old_str, new_str)"
        else:
            replace_method = "content.replace(old_str, new_str, 1)"

        lines = [
            f"# Edit {file_path}",
            f"python3 << 'PYTHON_EOF'",
            f"file_path = {repr(file_path)}",
            f"old_str = {old_escaped}",
            f"new_str = {new_escaped}",
            "",
            "with open(file_path, 'r') as f:",
            "    content = f.read()",
            "",
            f"content = {replace_method}",
            "",
            "with open(file_path, 'w') as f:",
            "    f.write(content)",
            "PYTHON_EOF",
        ]
        return lines

    def _handle_notebook_edit(self, step: TrajectoryStep) -> list[str]:
        """Handle NotebookEdit tool - Jupyter notebook editing via Python"""
        if not step.tool_input:
            return []

        notebook_path = step.tool_input.get("notebook_path", "")
        new_source = step.tool_input.get("new_source", "")
        cell_id = step.tool_input.get("cell_id")
        cell_type = step.tool_input.get("cell_type", "code")
        edit_mode = step.tool_input.get("edit_mode", "replace")

        if not notebook_path:
            return []

        new_source_escaped = self._escape_python_string(new_source)

        lines = [
            f"# Edit notebook {notebook_path}",
            f"python3 << 'PYTHON_EOF'",
            "import json",
            "",
            f"notebook_path = {repr(notebook_path)}",
            f"new_source = {new_source_escaped}",
            f"cell_id = {repr(cell_id)}",
            f"cell_type = {repr(cell_type)}",
            f"edit_mode = {repr(edit_mode)}",
            "",
            "with open(notebook_path, 'r') as f:",
            "    notebook = json.load(f)",
            "",
            "cells = notebook.get('cells', [])",
            "",
            "if edit_mode == 'insert':",
            "    new_cell = {",
            "        'cell_type': cell_type,",
            "        'source': new_source.split('\\n') if new_source else [],",
            "        'metadata': {},",
            "    }",
            "    if cell_type == 'code':",
            "        new_cell['execution_count'] = None",
            "        new_cell['outputs'] = []",
            "    # Insert at beginning if no cell_id, otherwise after the specified cell",
            "    if cell_id:",
            "        for i, cell in enumerate(cells):",
            "            if cell.get('id') == cell_id:",
            "                cells.insert(i + 1, new_cell)",
            "                break",
            "    else:",
            "        cells.insert(0, new_cell)",
            "elif edit_mode == 'delete':",
            "    if cell_id:",
            "        cells = [c for c in cells if c.get('id') != cell_id]",
            "else:  # replace",
            "    if cell_id:",
            "        for cell in cells:",
            "            if cell.get('id') == cell_id:",
            "                cell['source'] = new_source.split('\\n') if new_source else []",
            "                if cell_type:",
            "                    cell['cell_type'] = cell_type",
            "                break",
            "",
            "notebook['cells'] = cells",
            "",
            "with open(notebook_path, 'w') as f:",
            "    json.dump(notebook, f, indent=2)",
            "PYTHON_EOF",
        ]
        return lines

    def _escape_python_string(self, s: str) -> str:
        """Escape a string for use in Python code"""
        # Use triple quotes to handle multiline and special chars
        if "'''" in s and '"""' in s:
            # Fallback to repr for complex strings
            return repr(s)
        elif "'''" in s:
            return '"""' + s.replace('\\', '\\\\').replace('"', '\\"') + '"""'
        else:
            return "'''" + s.replace('\\', '\\\\') + "'''"

    def _sanitize_solve_sh(self, solve_sh: str, return_flag: bool = False) -> str | tuple[str, bool]:
        """
        Sanitize solve.sh to remove problematic imports from test files.

        This removes lines that import from expectation_tests.py or other test files,
        which would cause the solve.sh to fail in isolated verification environments.
        Handles both single-line and multi-line imports.

        Args:
            solve_sh: Raw solve.sh content
            return_flag: If True, return tuple (sanitized_content, had_imports_removed)

        Returns:
            Sanitized solve.sh content, or tuple if return_flag=True
        """
        # Patterns to detect start of test file imports
        test_import_start_patterns = [
            r'^from\s+expectation_tests\s+import\s+',
            r'^from\s+test_\w+\s+import\s+',
            r'^import\s+expectation_tests',
            r'^import\s+test_\w+',
        ]

        lines = solve_sh.split('\n')
        sanitized_lines = []
        removed_imports = []
        in_multiline_import = False
        paren_depth = 0

        for line in lines:
            stripped = line.strip()

            # Check if we're in a multi-line import
            if in_multiline_import:
                removed_imports.append(stripped)
                # Count parentheses to track when multi-line import ends
                paren_depth += stripped.count('(') - stripped.count(')')
                if paren_depth <= 0:
                    in_multiline_import = False
                    paren_depth = 0
                continue

            # Check if this line starts a test file import
            should_remove = False
            for pattern in test_import_start_patterns:
                if re.match(pattern, stripped):
                    should_remove = True
                    removed_imports.append(stripped)

                    # Check if it's a multi-line import (has opening paren but no closing)
                    open_parens = stripped.count('(')
                    close_parens = stripped.count(')')
                    if open_parens > close_parens:
                        in_multiline_import = True
                        paren_depth = open_parens - close_parens
                    break

            if not should_remove:
                sanitized_lines.append(line)

        had_imports_removed = len(removed_imports) > 0

        if had_imports_removed:
            print(f"[SolveShGenerator] Warning: Removed {len(removed_imports)} lines of test file imports from solve.sh:")
            for imp in removed_imports[:5]:  # Show first 5
                print(f"  - {imp}")
            if len(removed_imports) > 5:
                print(f"  ... and {len(removed_imports) - 5} more lines")

        sanitized_content = '\n'.join(sanitized_lines)

        if return_flag:
            return sanitized_content, had_imports_removed
        return sanitized_content

    def _extract_constants_from_tests(self, test_content: str) -> str:
        """
        Extract constant definitions from test file content.

        Looks for lines that define constants (UPPER_CASE = value) at module level.

        Args:
            test_content: Content of the test file

        Returns:
            String containing constant definitions
        """
        constants_lines = []
        lines = test_content.split('\n')

        # Patterns for constant definitions
        # Match: UPPER_CASE_NAME = value (at start of line, not indented)
        constant_pattern = r'^([A-Z][A-Z0-9_]*)\s*='

        in_multiline = False
        current_constant = []
        bracket_depth = 0

        for line in lines:
            stripped = line.strip()

            # Skip comments and empty lines
            if stripped.startswith('#') or not stripped:
                continue

            # Skip class and function definitions
            if stripped.startswith('class ') or stripped.startswith('def '):
                continue

            # Check if we're in a multiline constant
            if in_multiline:
                current_constant.append(line)
                bracket_depth += line.count('[') + line.count('{') + line.count('(')
                bracket_depth -= line.count(']') + line.count('}') + line.count(')')
                if bracket_depth <= 0:
                    constants_lines.extend(current_constant)
                    constants_lines.append('')
                    current_constant = []
                    in_multiline = False
                continue

            # Check for new constant definition (not indented)
            if re.match(constant_pattern, line) and not line.startswith(' ') and not line.startswith('\t'):
                # Check if it's a multiline definition
                bracket_depth = line.count('[') + line.count('{') + line.count('(')
                bracket_depth -= line.count(']') + line.count('}') + line.count(')')

                if bracket_depth > 0:
                    # Multiline constant
                    in_multiline = True
                    current_constant = [line]
                else:
                    # Single line constant
                    constants_lines.append(line)

        return '\n'.join(constants_lines)

    def regenerate_with_llm(
        self,
        trajectory: Trajectory,
        task_instruction: str,
        test_failures: list[TestFailure],
        previous_solve_sh: str,
        expectation_tests_content: str | None = None,
    ) -> str:
        """
        Use LLM to regenerate solve.sh when trajectory-based generation fails.

        Args:
            trajectory: Execution trajectory
            task_instruction: Task instruction content
            test_failures: List of test failures from verification
            previous_solve_sh: Previously failed solve.sh content
            expectation_tests_content: Content of expectation_tests.py (for extracting constants)

        Returns:
            Regenerated solve.sh content

        Raises:
            ValueError: If no LLM client is required for regeneration
        """
        if not self.llm_client:
            raise ValueError("LLM client is required for regeneration")

        # Generate trajectory summary
        trajectory_summary = self.trajectory_processor.summarize_for_llm(trajectory)

        # Format test failures
        failure_text = self._format_test_failures(test_failures)

        # Build constants section if test content is available
        constants_section = ""
        if expectation_tests_content:
            # Extract constants from test file
            constants_content = self._extract_constants_from_tests(expectation_tests_content)
            if constants_content:
                constants_section = CONSTANTS_REFERENCE_SECTION.format(
                    constants_content=constants_content,
                )

        # Build prompt
        user_prompt = REGENERATE_SOLVE_SH_USER.format(
            task_instruction=task_instruction,
            trajectory_summary=trajectory_summary,
            test_failures=failure_text,
            previous_solve_sh=previous_solve_sh,
            constants_section=constants_section,
        )

        # Call LLM
        response = self.llm_client.generate(
            system_prompt=REGENERATE_SOLVE_SH_SYSTEM,
            user_prompt=user_prompt,
            temperature=0.3,
        )

        # Clean up response and sanitize
        solve_sh = self._clean_solve_sh_response(response)
        solve_sh = self._sanitize_solve_sh(solve_sh)

        return solve_sh

    def generate_with_llm(
        self,
        trajectory: Trajectory,
        task_instruction: str,
    ) -> str:
        """
        Use LLM to generate solve.sh from scratch.

        Args:
            trajectory: Execution trajectory
            task_instruction: Task instruction content

        Returns:
            Generated solve.sh content

        Raises:
            ValueError: If no LLM client is configured
        """
        if not self.llm_client:
            raise ValueError("LLM client is required for generation")

        # Generate trajectory summary
        trajectory_summary = self.trajectory_processor.summarize_for_llm(trajectory)

        # Format output files
        output_files = "\n".join(f"- {f}" for f in trajectory.output_files)

        # Build prompt
        user_prompt = GENERATE_SOLVE_SH_USER.format(
            task_instruction=task_instruction,
            trajectory_summary=trajectory_summary,
            output_files=output_files,
        )

        # Call LLM
        response = self.llm_client.generate(
            system_prompt=GENERATE_SOLVE_SH_SYSTEM,
            user_prompt=user_prompt,
            temperature=0.3,
        )

        # Clean up response and sanitize
        solve_sh = self._clean_solve_sh_response(response)
        solve_sh = self._sanitize_solve_sh(solve_sh)

        return solve_sh

    # =========================================================================
    # Claude Code executor-based solve.sh generation
    # =========================================================================

    def generate_with_executor(
        self,
        trajectory: Trajectory,
        task_instruction: str,
        input_files: list[str],
        output_files: list[str],
        executor: "ClaudeExecutor",
        working_dir: str,
        solve_sh_path: str,
    ) -> str:
        """
        Generate solve.sh using Claude Code executor.

        Claude Code can read actual input/output files to generate more accurate scripts.

        Args:
            trajectory: Execution trajectory
            task_instruction: Task instruction content
            input_files: List of input file paths
            output_files: List of output file paths
            executor: Claude Code executor
            working_dir: Working directory for execution
            solve_sh_path: Path to write solve.sh

        Returns:
            Generated solve.sh content
        """
        # Build detailed trajectory info (including tool_output)
        trajectory_details = self._format_trajectory_with_outputs(trajectory)

        # Use the new method with trajectory summary
        return self.generate_with_trajectory_summary(
            task_instruction=task_instruction,
            trajectory_summary=trajectory_details,
            input_files=input_files,
            executor=executor,
            working_dir=working_dir,
            solve_sh_path=solve_sh_path,
        )

    def generate_with_trajectory_summary(
        self,
        task_instruction: str,
        trajectory_summary: str,
        input_files: list[str],
        executor: "ClaudeExecutor",
        working_dir: str,
        solve_sh_path: str,
    ) -> str:
        """
        Generate solve.sh using Claude Code executor with trajectory summary.

        This is the preferred method for the new pipeline that uses pre-processed
        trajectory summaries (with truncated tool_response).

        Args:
            task_instruction: Task instruction content
            trajectory_summary: Pre-processed trajectory summary (truncated responses)
            input_files: List of input file paths
            executor: Claude Code executor
            working_dir: Working directory for execution
            solve_sh_path: Path to write solve.sh

        Returns:
            Generated solve.sh content
        """
        from pathlib import Path

        input_files_list = "\n".join([f"- {f}" for f in input_files])

        prompt = GENERATE_SOLVE_WITH_EXECUTOR_USER.format(
            task_instruction=task_instruction,
            trajectory_summary=trajectory_summary,
            input_files_list=input_files_list,
            solve_sh_path=solve_sh_path,
        )

        full_prompt = f"{GENERATE_SOLVE_WITH_EXECUTOR_SYSTEM}\n\n---\n\n{prompt}"

        print(f"[SolveShGenerator] Using Claude Code to generate solve.sh...")

        result = executor.execute(
            prompt=full_prompt,
            working_dir=working_dir,
        )

        if not result.success:
            print(f"[SolveShGenerator] Claude Code execution failed: {result.error}")
            raise RuntimeError(f"Failed to generate solve.sh: {result.error}")

        # Read generated solve.sh
        solve_path = Path(solve_sh_path)
        if solve_path.exists():
            content = solve_path.read_text(encoding="utf-8")
            print(f"[SolveShGenerator] solve.sh generated: {len(content)} chars")
            # Sanitize
            return self._sanitize_solve_sh(content)

        raise RuntimeError("solve.sh file was not created")

    def refine_with_executor(
        self,
        previous_solve_sh: str,
        test_failures: list[TestFailure],
        executor: "ClaudeExecutor",
        working_dir: str,
        solve_sh_path: str,
        task_instruction: str | None = None,
        trajectory_summary: str | None = None,
    ) -> str:
        """
        Refine solve.sh using Claude Code based on test failures.

        Args:
            previous_solve_sh: Previously failed solve.sh content
            test_failures: List of test failures
            executor: Claude Code executor
            working_dir: Working directory
            solve_sh_path: Path to write refined solve.sh
            task_instruction: Task instruction (optional, for context)
            trajectory_summary: Trajectory summary (optional, for context)

        Returns:
            Refined solve.sh content
        """
        from pathlib import Path

        failure_text = self._format_test_failures(test_failures)

        prompt = REFINE_SOLVE_WITH_EXECUTOR_PROMPT.format(
            test_failures=failure_text,
            previous_solve_sh=previous_solve_sh,
            solve_sh_path=solve_sh_path,
            task_instruction=task_instruction or "(not provided)",
            trajectory_summary=trajectory_summary or "(not provided)",
        )

        print(f"[SolveShGenerator] Refining solve.sh with Claude Code...")

        result = executor.execute(
            prompt=prompt,
            working_dir=working_dir,
        )

        if not result.success:
            print(f"[SolveShGenerator] Refinement failed: {result.error}")
            return previous_solve_sh

        solve_path = Path(solve_sh_path)
        if solve_path.exists():
            content = solve_path.read_text(encoding="utf-8")
            print(f"[SolveShGenerator] Refined solve.sh: {len(content)} chars")
            return self._sanitize_solve_sh(content)

        return previous_solve_sh

    def _format_trajectory_with_outputs(self, trajectory: Trajectory) -> str:
        """
        Format trajectory with truncated tool_output for LLM prompt.

        Args:
            trajectory: Trajectory to format

        Returns:
            Formatted trajectory string
        """
        lines = []

        for step in trajectory.steps:
            if step.action_type != "tool_use":
                continue

            lines.append(f"### Step {step.step_id}: {step.tool_name}")

            if step.tool_input:
                # Format input based on tool type
                if step.tool_name == "Write":
                    lines.append(f"- file_path: {step.tool_input.get('file_path', '')}")
                    content = step.tool_input.get('content', '')
                    if len(content) > 500:
                        content = content[:500] + "... [truncated]"
                    lines.append(f"- content preview: {content[:200]}...")
                elif step.tool_name == "Bash":
                    lines.append(f"- command: {step.tool_input.get('command', '')}")
                elif step.tool_name == "Read":
                    lines.append(f"- file_path: {step.tool_input.get('file_path', '')}")
                elif step.tool_name == "Edit":
                    lines.append(f"- file_path: {step.tool_input.get('file_path', '')}")
                    lines.append(f"- old_string: {str(step.tool_input.get('old_string', ''))[:100]}...")
                    lines.append(f"- new_string: {str(step.tool_input.get('new_string', ''))[:100]}...")
                else:
                    # Generic format
                    input_str = str(step.tool_input)
                    if len(input_str) > 200:
                        input_str = input_str[:200] + "..."
                    lines.append(f"- input: {input_str}")

            if step.tool_output:
                # Include truncated output
                output_preview = step.tool_output
                if len(output_preview) > 300:
                    output_preview = output_preview[:300] + "... [truncated]"
                lines.append(f"- output: {output_preview}")

            lines.append("")

        return "\n".join(lines)

    def _format_test_failures(self, failures: list[TestFailure]) -> str:
        """Format test failures for LLM prompt"""
        if not failures:
            return "No specific test failures provided."

        lines = []
        for i, failure in enumerate(failures, 1):
            lines.append(f"### Failure {i}: {failure.test_name}")
            lines.append(f"Message: {failure.error_message}")
            if failure.traceback:
                lines.append(f"Details: {failure.traceback[:500]}")
            lines.append("")

        return "\n".join(lines)

    def _clean_solve_sh_response(self, response: str) -> str:
        """Clean LLM response to extract solve.sh content"""
        content = response.strip()

        # Remove markdown code blocks if present
        if content.startswith("```bash"):
            content = content[7:]
        elif content.startswith("```shell"):
            content = content[8:]
        elif content.startswith("```"):
            content = content[3:]

        if content.endswith("```"):
            content = content[:-3]

        content = content.strip()

        # Ensure shebang
        if not content.startswith("#!/bin/bash"):
            content = "#!/bin/bash\nset -e\n\n" + content

        return content
