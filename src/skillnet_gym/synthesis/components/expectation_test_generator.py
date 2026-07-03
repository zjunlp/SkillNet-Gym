"""Expectation test generator - generates tests BEFORE oracle execution.

This module implements Phase A of the two-phase test generation:
- Phase A (this): Generate tests based on instruction (before execution)
- Phase B (test_generator.py): Enhance tests based on actual output (after execution)
"""

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import FileSummaryResult

from ..prompts.expectation_test_prompts import (
    FIX_SYNTAX_ERROR_SYSTEM,
    FIX_SYNTAX_ERROR_USER,
    GENERATE_TESTS_SYSTEM,
    GENERATE_TESTS_USER,
    GENERATE_TESTS_WITH_EXECUTOR_SYSTEM,
    GENERATE_TESTS_WITH_EXECUTOR_USER,
    MERGE_TESTS_TEMPLATE,
    PROPOSER_REFINE_PROMPT,
    PROPOSER_REVIEW_SYSTEM,
    PROPOSER_REVIEW_USER,
    TEST_REVIEW_SYSTEM,
    TEST_REVIEW_USER,
    VERIFICATION_CHECKLIST_SYSTEM,
    VERIFICATION_CHECKLIST_USER,
    format_file_summary,
)
from ..utils.llm_client import LLMClient

# Type hint for executor
from typing import TYPE_CHECKING as _TYPE_CHECKING
if _TYPE_CHECKING:
    from ..execution.claude_executor import ClaudeExecutor


class ExpectationTestGenerator:
    """Generates expectation tests based on task instruction (before oracle execution).

    The generated tests verify that the oracle execution produced outputs
    that match the task instruction requirements, without depending on
    the actual output content.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        proposer_client: LLMClient | None = None,
    ):
        """
        Initialize the generator.

        Args:
            llm_client: LLM client for test generation
            proposer_client: Optional LLM client for proposer review (e.g., GPT-5.2)
        """
        self.llm = llm_client
        self.proposer = proposer_client

    def generate(
        self,
        task_instruction: str,
        exploration_summary: str,
        file_summary: "FileSummaryResult | None" = None,
        output_dir: str = "/root/output",
    ) -> str:
        """
        Generate expectation tests based on task instruction.

        Args:
            task_instruction: The task instruction to verify
            exploration_summary: Exploration report with skill capabilities
            file_summary: Input file summary (optional)
            output_dir: Expected output directory

        Returns:
            Complete pytest test file content
        """
        # Step 1: Extract verification checklist
        print("[ExpectationTestGenerator] Extracting verification checklist...")
        checklist = self._extract_verification_checklist(
            instruction=task_instruction,
            exploration_summary=exploration_summary,
            file_summary=file_summary,
        )
        print(f"[ExpectationTestGenerator] Checklist: {len(checklist.get('output_files', []))} files, "
              f"{len(checklist.get('required_content', []))} content checks")

        # Step 2: Generate tests from checklist
        print("[ExpectationTestGenerator] Generating tests from checklist...")
        tests = self._generate_tests_from_checklist(checklist, output_dir)

        # Step 3: Review and enhance tests
        print("[ExpectationTestGenerator] Reviewing and enhancing tests...")
        tests = self._review_and_enhance(task_instruction, tests)

        return tests

    def _extract_verification_checklist(
        self,
        instruction: str,
        exploration_summary: str,
        file_summary: "FileSummaryResult | None" = None,
    ) -> dict:
        """
        Extract verification checklist from instruction.

        Args:
            instruction: Task instruction
            exploration_summary: Exploration report
            file_summary: Input file summary

        Returns:
            Verification checklist dict
        """
        file_summary_text = format_file_summary(file_summary)

        prompt = VERIFICATION_CHECKLIST_USER.format(
            instruction=instruction,
            exploration_summary=exploration_summary,
            file_summary=file_summary_text,
        )

        response = self.llm.generate(
            system_prompt=VERIFICATION_CHECKLIST_SYSTEM,
            user_prompt=prompt,
            temperature=0.3,
        )

        # Parse JSON response
        checklist = self._parse_json_response(response)

        # Validate and fill defaults
        if not checklist:
            checklist = self._create_default_checklist(instruction)

        return checklist

    def _generate_tests_from_checklist(
        self,
        checklist: dict,
        output_dir: str,
    ) -> str:
        """
        Generate pytest code from verification checklist.

        Args:
            checklist: Verification checklist
            output_dir: Output directory path

        Returns:
            pytest test code string
        """
        prompt = GENERATE_TESTS_USER.format(
            checklist_json=json.dumps(checklist, indent=2, ensure_ascii=False),
            output_dir=output_dir,
        )

        response = self.llm.generate(
            system_prompt=GENERATE_TESTS_SYSTEM,
            user_prompt=prompt,
            temperature=0.3,
        )

        # Extract code from response
        tests = self._extract_code_from_response(response)

        # Validate it's valid Python
        if not self._is_valid_python(tests):
            # Fallback: generate minimal tests
            tests = self._generate_minimal_tests(checklist, output_dir)

        return tests

    def _review_and_enhance(
        self,
        instruction: str,
        generated_tests: str,
    ) -> str:
        """
        Review tests for completeness and enhance if needed.

        Args:
            instruction: Original task instruction
            generated_tests: Generated test code

        Returns:
            Enhanced test code
        """
        prompt = TEST_REVIEW_USER.format(
            instruction=instruction,
            generated_tests=generated_tests,
        )

        response = self.llm.generate(
            system_prompt=TEST_REVIEW_SYSTEM,
            user_prompt=prompt,
            temperature=0.3,
        )

        # Parse review result
        review = self._parse_json_response(response)

        if not review:
            return generated_tests

        # If coverage is complete, return as-is
        if review.get("coverage_complete", True):
            return generated_tests

        # Add missing tests
        missing_tests = review.get("missing_tests", [])
        if missing_tests:
            generated_tests = self._add_missing_tests(generated_tests, missing_tests)

        return generated_tests

    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from LLM response."""
        # Try to find JSON block
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try direct parse
            json_str = response.strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try to extract just the JSON part
            start = json_str.find('{')
            end = json_str.rfind('}') + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(json_str[start:end])
                except json.JSONDecodeError:
                    pass
            return {}

    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Try to find code block
        code_match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
        if code_match:
            return code_match.group(1)

        # Try plain code block
        code_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
        if code_match:
            return code_match.group(1)

        # Assume entire response is code
        return response.strip()

    def _is_valid_python(self, code: str) -> bool:
        """Check if code is valid Python syntax."""
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False

    def _create_default_checklist(self, instruction: str) -> dict:
        """Create a default checklist when LLM fails."""
        # Extract potential output file paths from instruction
        file_patterns = re.findall(r'/root/output/[\w.-]+', instruction)
        if not file_patterns:
            file_patterns = re.findall(r'/[\w/]+\.(?:json|csv|txt|md|xlsx)', instruction)

        output_files = []
        for path in file_patterns:
            ext = path.rsplit('.', 1)[-1] if '.' in path else 'txt'
            output_files.append({
                "path": path,
                "format": ext,
                "description": f"Output file extracted from instruction",
            })

        # If no files found, add a generic one
        if not output_files:
            output_files.append({
                "path": "/root/output/result.json",
                "format": "json",
                "description": "Default output file",
            })

        return {
            "output_files": output_files,
            "required_content": [
                {
                    "file": output_files[0]["path"].split("/")[-1],
                    "check_type": "not_empty",
                    "params": {},
                    "description": "Output should not be empty",
                }
            ],
            "data_constraints": [],
            "semantic_checks": [],
        }

    def _generate_minimal_tests(self, checklist: dict, output_dir: str) -> str:
        """Generate minimal tests when LLM fails."""
        test_methods = []

        # Generate file existence tests
        for i, file_info in enumerate(checklist.get("output_files", [])):
            path = file_info.get("path", f"{output_dir}/output_{i}.txt")
            fmt = file_info.get("format", "txt")
            test_name = f"output_{i}_{fmt}"

            test_methods.append(f'''
    def test_{test_name}_exists(self):
        """Verify {path} was created."""
        assert os.path.exists("{path}"), f"Output file not found: {path}"
''')

            # Add format validation for JSON/CSV
            if fmt == "json":
                test_methods.append(f'''
    def test_{test_name}_valid_json(self):
        """Verify {path} is valid JSON."""
        with open("{path}") as f:
            data = json.load(f)
        assert data is not None, "JSON file is empty or null"
''')
            elif fmt == "csv":
                test_methods.append(f'''
    def test_{test_name}_has_data(self):
        """Verify {path} has data rows."""
        with open("{path}") as f:
            lines = f.readlines()
        assert len(lines) > 1, "CSV file has no data rows"
''')

        # Generate content checks
        for check in checklist.get("required_content", []):
            check_type = check.get("check_type", "not_empty")
            file_name = check.get("file", "output.txt")
            desc = check.get("description", "Content check")

            # Find full path
            full_path = f"{output_dir}/{file_name}"
            for f in checklist.get("output_files", []):
                if f.get("path", "").endswith(file_name):
                    full_path = f.get("path")
                    break

            test_name = f"{file_name.replace('.', '_')}_{check_type}"

            if check_type == "not_empty":
                test_methods.append(f'''
    def test_{test_name}(self):
        """{desc}"""
        with open("{full_path}") as f:
            content = f.read().strip()
        assert len(content) > 0, "File is empty"
''')
            elif check_type == "contains_key":
                key = check.get("params", {}).get("key", "data")
                test_methods.append(f'''
    def test_{test_name}(self):
        """{desc}"""
        with open("{full_path}") as f:
            data = json.load(f)
        assert "{key}" in data, "Missing required key: {key}"
''')

        if not test_methods:
            test_methods.append('''
    def test_output_exists(self):
        """Basic test: check output directory has files."""
        output_dir = Path("/root/output")
        if output_dir.is_dir():
            files = list(output_dir.iterdir())
            assert len(files) > 0, "No output files produced"
        else:
            pytest.fail("Output directory not found")
''')

        return MERGE_TESTS_TEMPLATE.format(test_methods="\n".join(test_methods))

    def _add_missing_tests(self, existing_tests: str, missing_tests: list) -> str:
        """Add missing tests to existing test code."""
        # Find the class body end
        class_end = existing_tests.rfind('\n\n')
        if class_end == -1:
            class_end = len(existing_tests)

        # Add missing test methods
        new_methods = []
        for test in missing_tests:
            code = test.get("code", "")
            if code and "def test_" in code:
                # Ensure proper indentation
                lines = code.split('\n')
                indented_lines = []
                for line in lines:
                    if line.strip():
                        if not line.startswith('    '):
                            line = '    ' + line
                        indented_lines.append(line)
                    else:
                        indented_lines.append(line)
                new_methods.append('\n'.join(indented_lines))

        if new_methods:
            # Insert before class end
            insert_point = existing_tests.rfind('\n', 0, class_end)
            if insert_point == -1:
                insert_point = class_end

            return (
                existing_tests[:insert_point] +
                '\n' +
                '\n'.join(new_methods) +
                existing_tests[insert_point:]
            )

        return existing_tests

    # =========================================================================
    # Claude Code Executor 方式生成测试
    # =========================================================================

    def generate_with_executor(
        self,
        task_instruction: str,
        exploration_summary: str,
        input_files: list[str],
        output_dir: str,
        executor: "ClaudeExecutor",
        working_dir: str,
        test_file_path: str | None = None,
        enable_proposer: bool = True,
        max_proposer_iterations: int = 2,
    ) -> str:
        """
        使用 Claude Code 执行器生成测试。

        这种方式允许 Claude Code 实际读取输入文件，
        了解其结构后生成更精确的验证测试。

        Args:
            task_instruction: 任务指令
            exploration_summary: 探索报告
            input_files: 输入文件路径列表
            output_dir: 输出目录
            executor: Claude Code 执行器
            working_dir: 工作目录
            test_file_path: 测试文件保存路径（可选）
            enable_proposer: 是否启用 proposer 审查
            max_proposer_iterations: proposer 最大交互次数

        Returns:
            生成的测试代码
        """
        from pathlib import Path

        # 确定测试文件路径
        if test_file_path is None:
            test_file_path = str(Path(working_dir) / "expectation_tests.py")

        # 构建输入文件列表
        input_files_list = "\n".join([f"- {path}" for path in input_files])

        # 构建 prompt（将系统指令合并到用户 prompt 中）
        user_prompt = GENERATE_TESTS_WITH_EXECUTOR_USER.format(
            instruction=task_instruction,
            exploration_summary=exploration_summary,
            input_files_list=input_files_list,
            output_dir=output_dir,
            test_file_path=test_file_path,
        )

        # 合并系统提示和用户提示
        full_prompt = f"{GENERATE_TESTS_WITH_EXECUTOR_SYSTEM}\n\n---\n\n{user_prompt}"

        print(f"[ExpectationTestGenerator] Using Claude Code to generate tests...")
        print(f"[ExpectationTestGenerator] Input files: {len(input_files)}")

        # 使用执行器生成测试
        result = executor.execute(
            prompt=full_prompt,
            working_dir=working_dir,
        )

        # 检查执行结果
        if not result.success:
            print(f"[ExpectationTestGenerator] Claude Code execution failed: {result.error}")
            # 回退到 LLM 方式
            print(f"[ExpectationTestGenerator] Falling back to LLM-only generation...")
            return self.generate(
                task_instruction=task_instruction,
                exploration_summary=exploration_summary,
                output_dir=output_dir,
            )

        # 读取生成的测试文件
        test_path = Path(test_file_path)
        if not test_path.exists():
            print(f"[ExpectationTestGenerator] Test file not created, falling back...")
            return self.generate(
                task_instruction=task_instruction,
                exploration_summary=exploration_summary,
                output_dir=output_dir,
            )

        tests = test_path.read_text(encoding="utf-8")
        print(f"[ExpectationTestGenerator] Tests generated: {len(tests)} chars")

        # Proposer 审查循环（如果启用且有 proposer client）
        if enable_proposer and self.proposer:
            tests = self._proposer_review_loop(
                tests=tests,
                task_instruction=task_instruction,
                input_files=input_files,
                executor=executor,
                working_dir=working_dir,
                test_file_path=test_file_path,
                max_iterations=max_proposer_iterations,
            )
            # 审查后重新写入文件
            test_path.write_text(tests, encoding="utf-8")

        # 验证并修复语法错误
        tests = self._validate_and_fix_syntax(tests)

        return tests

    def _validate_and_fix_syntax(
        self,
        code: str,
        max_retries: int = 2,
    ) -> str:
        """
        验证代码语法，如果有错误则尝试修复。

        Args:
            code: Python 代码
            max_retries: 最大修复重试次数

        Returns:
            修复后的代码（或原始代码如果无法修复）
        """
        for attempt in range(max_retries + 1):
            is_valid, error = self._validate_syntax(code)

            if is_valid:
                if attempt > 0:
                    print(f"[ExpectationTestGenerator] Syntax fixed after {attempt} attempt(s)")
                return code

            if attempt < max_retries:
                print(f"[ExpectationTestGenerator] Syntax error: {error}")
                print(f"[ExpectationTestGenerator] Attempting fix ({attempt + 1}/{max_retries})...")
                code = self._fix_syntax_errors(code, error)

        # 最终仍有错误，返回最后一次修复的结果
        print(f"[ExpectationTestGenerator] Warning: Could not fully fix syntax errors")
        return code

    def _validate_syntax(self, code: str) -> tuple[bool, str]:
        """
        验证 Python 代码语法。

        Args:
            code: Python 代码

        Returns:
            (是否有效, 错误信息)
        """
        try:
            compile(code, '<string>', 'exec')
            return True, ""
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"

    def _fix_syntax_errors(self, code: str, error: str) -> str:
        """
        使用 LLM 修复语法错误。

        Args:
            code: 有语法错误的代码
            error: 错误信息

        Returns:
            修复后的代码
        """
        prompt = FIX_SYNTAX_ERROR_USER.format(code=code, error=error)

        response = self.llm.generate(
            system_prompt=FIX_SYNTAX_ERROR_SYSTEM,
            user_prompt=prompt,
            temperature=0.1,
        )

        # 提取代码
        fixed_code = self._extract_code_from_response(response)

        # 如果提取失败，返回原始代码
        if not fixed_code or len(fixed_code) < 50:
            return code

        return fixed_code

    # =========================================================================
    # Proposer 审查方法 (Phase 3.3.5)
    # =========================================================================

    def _proposer_review_loop(
        self,
        tests: str,
        task_instruction: str,
        input_files: list[str],
        executor: "ClaudeExecutor",
        working_dir: str,
        test_file_path: str,
        max_iterations: int = 2,
    ) -> str:
        """
        Proposer 审查循环。

        如果 proposer 发现问题，让 Claude Code 改进测试，
        直到 proposer 批准或达到最大迭代次数。

        Args:
            tests: 当前测试代码
            task_instruction: 任务指令
            input_files: 输入文件列表
            executor: Claude Code 执行器
            working_dir: 工作目录
            test_file_path: 测试文件路径
            max_iterations: 最大迭代次数

        Returns:
            改进后的测试代码
        """
        from pathlib import Path

        for iteration in range(max_iterations):
            print(f"[ExpectationTestGenerator] Proposer review iteration {iteration + 1}/{max_iterations}...")

            # 调用 Proposer 审查
            review = self._proposer_review(tests, task_instruction, input_files)

            if review.get("approved", False):
                print(f"[ExpectationTestGenerator] Proposer approved tests!")
                return tests

            # Proposer 发现问题，让 Claude Code 改进
            summary = review.get("summary", "Issues found")
            print(f"[ExpectationTestGenerator] Proposer found issues: {summary}")

            feedback = self._format_proposer_feedback(review)
            tests = self._refine_with_executor(
                executor=executor,
                working_dir=working_dir,
                test_file_path=test_file_path,
                previous_tests=tests,
                proposer_feedback=feedback,
            )

        # 达到最大迭代次数，返回最后版本
        print(f"[ExpectationTestGenerator] Max proposer iterations reached, using last version")
        return tests

    def _proposer_review(
        self,
        tests: str,
        task_instruction: str,
        input_files: list[str],
    ) -> dict:
        """
        调用 Proposer LLM 审查测试。

        Args:
            tests: 测试代码
            task_instruction: 任务指令
            input_files: 输入文件列表

        Returns:
            审查结果字典，包含 approved, issues, summary 字段
        """
        input_files_summary = "\n".join([f"- {f}" for f in input_files])

        prompt = PROPOSER_REVIEW_USER.format(
            task_instruction=task_instruction,
            input_files_summary=input_files_summary,
            generated_tests=tests,
        )

        response = self.proposer.generate(
            system_prompt=PROPOSER_REVIEW_SYSTEM,
            user_prompt=prompt,
            temperature=0.2,
        )

        return self._parse_json_response(response)

    def _format_proposer_feedback(self, review: dict) -> str:
        """
        格式化 Proposer 审查结果为反馈文本。

        Args:
            review: Proposer 审查结果

        Returns:
            格式化的反馈文本
        """
        lines = []

        summary = review.get("summary", "")
        if summary:
            lines.append(f"Summary: {summary}")
            lines.append("")

        issues = review.get("issues", [])
        if issues:
            lines.append("Issues to address:")
            for i, issue in enumerate(issues, 1):
                category = issue.get("category", "unknown")
                description = issue.get("description", "")
                suggestion = issue.get("suggestion", "")

                lines.append(f"\n{i}. [{category.upper()}] {description}")
                if suggestion:
                    lines.append(f"   Suggestion: {suggestion}")

        return "\n".join(lines)

    def _refine_with_executor(
        self,
        executor: "ClaudeExecutor",
        working_dir: str,
        test_file_path: str,
        previous_tests: str,
        proposer_feedback: str,
    ) -> str:
        """
        让 Claude Code 根据 Proposer 反馈改进测试。

        Args:
            executor: Claude Code 执行器
            working_dir: 工作目录
            test_file_path: 测试文件路径
            previous_tests: 之前的测试代码
            proposer_feedback: Proposer 反馈

        Returns:
            改进后的测试代码
        """
        from pathlib import Path

        refine_prompt = PROPOSER_REFINE_PROMPT.format(
            proposer_feedback=proposer_feedback,
            previous_tests=previous_tests,
            test_file_path=test_file_path,
        )

        print(f"[ExpectationTestGenerator] Refining tests with Claude Code...")

        result = executor.execute(
            prompt=refine_prompt,
            working_dir=working_dir,
        )

        if not result.success:
            print(f"[ExpectationTestGenerator] Refinement failed: {result.error}")
            return previous_tests

        # 读取改进后的测试
        test_path = Path(test_file_path)
        if test_path.exists():
            refined_tests = test_path.read_text(encoding="utf-8")
            print(f"[ExpectationTestGenerator] Refined tests: {len(refined_tests)} chars")
            return refined_tests

        return previous_tests
