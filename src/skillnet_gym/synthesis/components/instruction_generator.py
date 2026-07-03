"""Instruction generator for creating task descriptions from trajectories"""

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import DAGTask, FileSummaryResult

from ..config import FileSummaryEntry, ProcessedTrajectory, Trajectory
from ..utils.llm_client import LLMClient
from .trajectory_processor import TrajectoryProcessor


# System prompt for instruction generation
SYSTEM_PROMPT = """You are a task instruction synthesis expert. Given an Agent's execution trajectory, you generate semantically consistent task instructions.

Your generated instructions must:
1. Clearly define input and output requirements
2. NOT include specific implementation steps or code
3. Be semantically consistent with the trajectory operations
4. Use clear, concise language
5. Specify output format requirements (JSON, CSV, MD, etc.)
6. Be challenging but achievable for an AI coding agent

Output format: Return ONLY the instruction content in markdown format, suitable for use as instruction.md"""


# Example instructions for few-shot learning
EXAMPLE_INSTRUCTIONS = [
    (
        # Example 1: Data recovery
        """Trajectory Summary:
- Read Excel file with missing values marked as "???"
- Analyzed relationships between sheets
- Calculated correct values
- Created new Excel file with recovered values

Input: nasa_budget_incomplete.xlsx
Output: nasa_budget_recovered.xlsx""",
        """Recover missing values in an Excel file `nasa_budget_incomplete.xlsx`. Missing values are marked with "???".

Analyze the relationships between sheets to determine the correct values and replace each "???" with the computed numeric value.

Save the result as `nasa_budget_recovered.xlsx` with all missing values filled in."""
    ),
    (
        # Example 2: Document analysis
        """Trajectory Summary:
- Read PDF contract document
- Extracted key information (parties, amounts, dates)
- Performed risk analysis
- Created multiple output files: analysis.md, key_points.json, risks.txt

Input: contract.pdf
Output: analysis.md, key_points.json, risks.txt""",
        """Analyze the contract document `/root/contract.pdf` and extract key information.

You need to:
1. Extract basic contract information (parties, amounts, dates, terms)
2. Identify potential risks and anomalies
3. Generate a structured analysis

Output files:
- `/root/output/analysis.md` - Comprehensive analysis report
- `/root/output/key_points.json` - Structured data with key contract points
- `/root/output/risks.txt` - Risk assessment summary"""
    ),
    (
        # Example 3: 3D calculation
        """Trajectory Summary:
- Parsed binary STL file
- Found largest connected component
- Extracted material ID
- Calculated mass using density table
- Output JSON with mass and material ID

Input: scan_data.stl, material_density_table.md
Output: mass_report.json""",
        """Calculate the mass of a 3D printed part. The input (`/root/scan_data.stl`) is a binary STL with Material ID stored in the attribute byte count field.

You need to:
1. Parse the binary STL and identify the **largest connected component** by filtering out any scanning debris
2. Extract the Material ID and reference it with `/root/material_density_table.md` to find the corresponding density
3. Calculate the mass using `Volume * Density`

Save the result to `/root/mass_report.json`:
```json
{
  "main_part_mass": 12345.67,
  "material_id": 42
}
```

NOTE: Result accuracy within 0.1% is acceptable."""
    ),
]


USER_PROMPT_TEMPLATE = """## Trajectory Summary
{trajectory_summary}

## Input Files
{input_files}

## Output Files
{output_files}

## File Types
{file_types}

---

Based on this trajectory, generate a task instruction that:
1. Clearly defines what needs to be done with the input file(s)
2. Specifies the required output format and structure
3. Does NOT include implementation details or step-by-step code
4. Is semantically consistent with the operations performed

Generate the instruction.md content:"""


class InstructionGenerator:
    """Generates task instructions from execution trajectories"""

    def __init__(self, llm_client: LLMClient | None = None):
        """
        Initialize the generator.

        Args:
            llm_client: LLM client for generation. Creates default if None.
        """
        self.llm = llm_client
        self.processor = TrajectoryProcessor()

    def generate(
        self,
        trajectory: Trajectory,
        input_files: list[str] | None = None,
        use_examples: bool = True,
    ) -> str:
        """
        Generate task instruction from trajectory.

        Args:
            trajectory: Execution trajectory to analyze
            input_files: Override input files list
            use_examples: Whether to include few-shot examples

        Returns:
            Generated instruction.md content
        """
        if self.llm is None:
            raise ValueError("LLM client is required for instruction generation")

        # Get trajectory summary
        summary = self.processor.summarize_for_llm(trajectory)

        # Prepare input/output file lists
        inputs = input_files or trajectory.input_files
        outputs = trajectory.output_files

        # Get file types
        processed = self.processor.process(trajectory)
        file_types = processed.metadata.get("file_types", [])

        # Build user prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(
            trajectory_summary=summary,
            input_files="\n".join(f"- {f}" for f in inputs) if inputs else "- (no input files)",
            output_files="\n".join(f"- {f}" for f in outputs) if outputs else "- (no output files specified)",
            file_types=", ".join(file_types) if file_types else "unknown",
        )

        # Generate instruction
        if use_examples:
            instruction = self.llm.generate_with_examples(
                system_prompt=SYSTEM_PROMPT,
                examples=EXAMPLE_INSTRUCTIONS,
                user_prompt=user_prompt,
                temperature=0.7,
            )
        else:
            instruction = self.llm.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.7,
            )

        return self._post_process(instruction, inputs)

    def generate_from_processed(
        self,
        processed: ProcessedTrajectory,
        use_examples: bool = True,
    ) -> str:
        """
        Generate instruction from a ProcessedTrajectory.

        Args:
            processed: Processed trajectory
            use_examples: Whether to include few-shot examples

        Returns:
            Generated instruction.md content
        """
        return self.generate(
            trajectory=processed.trajectory,
            input_files=None,
            use_examples=use_examples,
        )

    def _post_process(self, instruction: str, input_files: list[str] | None = None) -> str:
        """Clean up and format the generated instruction.

        Note: Full path normalization is handled by PathNormalizer in TaskPackager.
        This method only performs markdown formatting cleanup.
        """
        # Remove any markdown code fences if present
        instruction = instruction.strip()
        if instruction.startswith("```markdown"):
            instruction = instruction[len("```markdown"):].strip()
        if instruction.startswith("```md"):
            instruction = instruction[len("```md"):].strip()
        if instruction.startswith("```"):
            instruction = instruction[3:].strip()
        if instruction.endswith("```"):
            instruction = instruction[:-3].strip()

        # Path normalization is now handled by PathNormalizer in TaskPackager
        # to ensure consistent normalization across all output artifacts
        # (instruction, tests, solve.sh)

        return instruction

    def generate_simple(
        self,
        input_file: str | None,
        output_files: list[str],
        operation_summary: str,
    ) -> str:
        """
        Generate a simple instruction without full trajectory.

        Useful for quick generation when full trajectory isn't needed.

        Args:
            input_file: Input file path
            output_files: List of output file paths
            operation_summary: Brief description of operations performed

        Returns:
            Generated instruction content
        """
        if self.llm is None:
            raise ValueError("LLM client is required for instruction generation")

        prompt = f"""Generate a clear task instruction based on:

Input file: {input_file or "(none)"}
Output files: {", ".join(output_files)}
Operations: {operation_summary}

The instruction should:
1. Describe what to do with the input
2. Specify the required outputs
3. NOT include implementation steps

Generate the instruction:"""

        return self._post_process(self.llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.7,
        ))

    def validate_instruction(
        self,
        instruction: str,
        trajectory: Trajectory,
    ) -> tuple[bool, list[str]]:
        """
        Validate that an instruction is semantically consistent with trajectory.

        Args:
            instruction: Generated instruction
            trajectory: Original trajectory

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        instruction_lower = instruction.lower()

        # Check that output files are mentioned
        for output_file in trajectory.output_files:
            filename = output_file.split("/")[-1]
            if filename.lower() not in instruction_lower:
                issues.append(f"Output file '{filename}' not mentioned in instruction")

        # Check that input files are mentioned
        for input_file in trajectory.input_files:
            filename = input_file.split("/")[-1]
            if filename.lower() not in instruction_lower:
                issues.append(f"Input file '{filename}' not mentioned in instruction")

        # Check for implementation details (warning, not error)
        implementation_keywords = [
            "step 1:", "step 2:", "first,", "then,", "finally,",
            "```python", "```bash", "def ", "import ",
        ]
        for keyword in implementation_keywords:
            if keyword.lower() in instruction_lower:
                issues.append(f"Possible implementation detail: '{keyword}'")

        return len(issues) == 0, issues

    def verify_instruction_consistency(
        self,
        instruction: str,
        trajectory: Trajectory,
        output_files_content: dict[str, str],
    ) -> tuple[bool, str]:
        """
        使用 LLM 验证指令与轨迹/输出是否一致。

        Args:
            instruction: 任务指令
            trajectory: 执行轨迹
            output_files_content: 输出文件内容映射

        Returns:
            Tuple of (is_consistent, reason)
            - is_consistent: 是否一致
            - reason: 不一致的原因（一致时为空字符串）
        """
        if self.llm is None:
            raise ValueError("LLM client is required for consistency verification")

        # 获取轨迹摘要
        summary = self.processor.summarize_for_llm(trajectory)

        # 格式化输出文件内容（限制长度）
        outputs_text = ""
        for path, content in output_files_content.items():
            truncated = content[:2000] + "..." if len(content) > 2000 else content
            outputs_text += f"\n### {path}\n```\n{truncated}\n```\n"

        verify_prompt = f"""## Task Instruction
{instruction}

## Execution Trajectory Summary
{summary}

## Output Files Content
{outputs_text}

---

Analyze whether the execution trajectory and output files are consistent with the task instruction.

Consider:
1. Does the output match what the instruction asks for?
2. Are there any significant deviations from the instruction?
3. Are the output file formats and contents as expected?

Respond in the following format:
CONSISTENT: [YES/NO]
REASON: [If NO, explain the inconsistency. If YES, leave empty or say "Consistent"]
"""

        response = self.llm.generate(
            system_prompt="You are a task verification expert. Analyze whether an execution matches its instruction.",
            user_prompt=verify_prompt,
            temperature=0.3,
        )

        # 解析响应
        is_consistent = "CONSISTENT: YES" in response.upper() or "CONSISTENT:YES" in response.upper()

        # 提取原因
        reason = ""
        if "REASON:" in response.upper():
            reason_start = response.upper().find("REASON:") + len("REASON:")
            reason = response[reason_start:].strip()
            # 清理空的原因
            if reason.lower() in ["consistent", "none", "n/a", ""]:
                reason = ""

        return is_consistent, reason

    def refine_instruction(
        self,
        original_instruction: str,
        trajectory: Trajectory,
        inconsistency_reason: str,
        output_files_content: dict[str, str] | None = None,
    ) -> str:
        """
        基于轨迹修正指令，使其与实际执行一致。

        Args:
            original_instruction: 原始任务指令
            trajectory: 执行轨迹
            inconsistency_reason: 不一致的原因
            output_files_content: 输出文件内容（可选）

        Returns:
            修正后的指令
        """
        if self.llm is None:
            raise ValueError("LLM client is required for instruction refinement")

        # 获取轨迹摘要
        summary = self.processor.summarize_for_llm(trajectory)

        # 格式化输出文件
        outputs_text = ""
        if output_files_content:
            for path, content in output_files_content.items():
                truncated = content[:1000] + "..." if len(content) > 1000 else content
                outputs_text += f"\n- {path}: {truncated[:200]}...\n"

        refine_prompt = f"""## Original Instruction
{original_instruction}

## Inconsistency Reason
{inconsistency_reason}

## Actual Execution Summary
{summary}

## Actual Output Files
{outputs_text or "[No output content available]"}

---

The original instruction does not match what was actually executed.
Refine the instruction to accurately describe what was actually done and produced.

Requirements:
1. Keep the instruction clear and concise
2. Do NOT include implementation steps or code
3. Accurately reflect the actual input/output files
4. Specify the correct output format based on actual outputs

Generate the refined instruction:"""

        refined = self.llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=refine_prompt,
            temperature=0.5,
        )

        return self._post_process(refined, trajectory.input_files)

    def generate_task_from_exploration(
        self,
        exploration_summary: str,
        file_summaries: "FileSummaryResult",
        skills: list[str],
        use_metadata: bool = True,
    ) -> str:
        """
        使用 task_synthesize.py 的 prompt 从探索报告生成任务指令。

        Args:
            exploration_summary: 探索总结内容（exploration_summary.md 的内容）
            file_summaries: FileSummaryResult 对象，包含文件摘要和类型
            skills: 可用技能列表
            use_metadata: 是否使用详细 metadata 丰富上下文（默认 True）

        Returns:
            生成的任务指令
        """
        if self.llm is None:
            raise ValueError("LLM client is required for task generation")

        from ..prompts.task_synthesize import SYS_PROMPT as TASK_SYS_PROMPT
        from ..prompts.task_synthesize import USER_PROMPT as TASK_USER_PROMPT
        from ..prompts.task_synthesize import MULTI_FILE_INSTRUCTION
        from ..config import FileSummaryResult

        # 格式化可用技能
        skills_text = ", ".join(skills) if skills else "None specified"

        # 格式化可用文件（包含摘要、类型和可选的 metadata）
        files_text = ""
        file_count = len(file_summaries.files)
        for entry in file_summaries.files:
            if use_metadata and entry.metadata:
                files_text += self._format_file_with_metadata(entry)
            else:
                files_text += f"- **{entry.name}** ({entry.content_type})\n"
                files_text += f"  Path: {entry.path}\n"
                files_text += f"  Summary: {entry.summary}\n\n"

        if not files_text:
            files_text = "No files available"

        # 构建 prompt
        prompt = TASK_USER_PROMPT.format(
            summary_table=exploration_summary,
            skills=skills_text,
            files=files_text,
        )

        # 如果有多个文件，添加多文件任务指令
        if file_count > 1:
            multi_file_note = MULTI_FILE_INSTRUCTION.format(file_count=file_count)
            prompt = prompt + "\n" + multi_file_note

        print("="*80)
        print(f"[InstructionGenerator] Generating task for {file_count} file(s)")
        print("="*50)

        # 调用 LLM（使用 reasoning_effort="low" 加速合成，不指定 temperature）
        response = self.llm.generate(
            system_prompt=TASK_SYS_PROMPT,
            user_prompt=prompt,
            # reasoning_effort="low",
            temperature=0.6,
        )

        # 提取任务指令部分
        return self._extract_task_instruction(response)

    def _format_file_with_metadata(self, entry: FileSummaryEntry) -> str:
        """
        格式化包含 metadata 的文件信息。

        Args:
            entry: FileSummaryEntry 对象

        Returns:
            格式化的文件信息字符串
        """
        lines = [f"### {entry.name}"]
        lines.append(f"- **Content Type**: {entry.content_type}")
        lines.append(f"- **Path**: {entry.path}")
        lines.append(f"- **Summary**: {entry.summary}")

        if entry.metadata:
            lines.append("")
            lines.append("**Metadata**:")

            # 定义显示优先级的字段
            priority_fields = [
                "domain", "document_type", "language", "complexity_level",
                "page_count", "row_count", "column_count", "sheet_count",
                "column_names", "numeric_columns", "has_tables", "has_formulas",
                "field_count", "is_fillable", "potential_operations",
            ]

            # 先显示优先级字段
            shown_keys = set()
            for key in priority_fields:
                if key in entry.metadata:
                    value = self._format_metadata_value(entry.metadata[key])
                    lines.append(f"  - {key}: {value}")
                    shown_keys.add(key)

            # 显示剩余字段（排除 error 和过长的字段）
            skip_keys = {"extraction_error", "llm_extraction_error", "extraction_note", "file_size_bytes"}
            for key, value in entry.metadata.items():
                if key not in shown_keys and key not in skip_keys:
                    formatted = self._format_metadata_value(value)
                    if len(formatted) < 200:  # Skip overly long values
                        lines.append(f"  - {key}: {formatted}")

        lines.append("")  # Empty line after entry
        return "\n".join(lines) + "\n"

    def _format_metadata_value(self, value) -> str:
        """
        格式化 metadata 值为字符串。

        Args:
            value: 任意类型的值

        Returns:
            格式化的字符串
        """
        if isinstance(value, list):
            if len(value) > 5:
                return ", ".join(str(v) for v in value[:5]) + f"... (+{len(value) - 5} more)"
            return ", ".join(str(v) for v in value)
        elif isinstance(value, dict):
            # Compact JSON representation
            return json.dumps(value, ensure_ascii=False)[:150]
        elif isinstance(value, bool):
            return "yes" if value else "no"
        elif value is None:
            return "N/A"
        else:
            return str(value)

    def _extract_task_instruction(self, response: str) -> str:
        """
        从 LLM 响应中提取任务指令部分。

        Args:
            response: LLM 完整响应

        Returns:
            提取的任务指令
        """
        import re

        # 尝试匹配 "# Synthesized Task Instruction" 后的内容
        match = re.search(
            r'#\s*Synthesized\s*Task\s*Instruction\s*\n(.*)',
            response,
            re.DOTALL | re.IGNORECASE
        )

        if match:
            instruction = match.group(1).strip()
            # 移除可能的后续 section
            for section_marker in ["# ", "## ", "---"]:
                if section_marker in instruction:
                    instruction = instruction.split(section_marker)[0].strip()
            return instruction

        # Fallback: 返回整个响应
        return response.strip()

    def generate_guiding_metadata(
        self,
        task_instruction: str,
        exploration_summary: str,
        file_summaries: "FileSummaryResult",
        skills: list[str] | None = None,
        dag_task: "DAGTask | None" = None,
    ) -> str:
        """
        使用 guiding_metadata.py 的 prompt 生成引导思路。

        这是任务指令生成后、oracle轨迹生成前的步骤。

        Args:
            task_instruction: 已生成的任务指令
            exploration_summary: 探索总结内容
            file_summaries: FileSummaryResult 对象
            skills: 可用技能列表（可选）
            dag_task: Optional DAGTask for DAG workflow guidance (V2)

        Returns:
            生成的引导思路（guiding metadata）
        """
        if self.llm is None:
            raise ValueError("LLM client is required for guiding metadata generation")

        from ..prompts.guiding_metadata import SYS_PROMPT as GUIDE_SYS_PROMPT
        from ..prompts.guiding_metadata import USER_PROMPT as GUIDE_USER_PROMPT

        # 格式化可用技能
        skills_text = ", ".join(skills) if skills else "See exploration summary"

        # 格式化目标文件
        files_text = ""
        for entry in file_summaries.files:
            files_text += f"- **{entry.name}** (Content type: {entry.content_type})\n"
            files_text += f"  Path: {entry.path}\n"
            files_text += f"  Summary: {entry.summary}\n\n"

        if not files_text:
            files_text = "No target files specified"

        # 格式化 DAG workflow section (V2 only)
        dag_workflow_section = ""
        if dag_task is not None:
            dag_workflow_section = self._format_dag_workflow_section(dag_task)

        # 构建 prompt
        prompt = GUIDE_USER_PROMPT.format(
            task_instruction=task_instruction,
            exploration_summary=exploration_summary,
            skills=skills_text,
            files=files_text,
            dag_workflow_section=dag_workflow_section,
        )

        # 调用 LLM（使用 reasoning_effort="low" 加速合成，不指定 temperature）
        response = self.llm.generate(
            system_prompt=GUIDE_SYS_PROMPT,
            user_prompt=prompt,
            # reasoning_effort="low",
            temperature=0.6,
        )

        return self._extract_guiding_metadata(response)

    def _format_dag_workflow_section(self, dag_task: "DAGTask") -> str:
        """
        Format DAG workflow information as a prompt section.

        Args:
            dag_task: DAGTask with DAG structure info

        Returns:
            Formatted DAG workflow section string
        """
        topo_order = " -> ".join(dag_task.topological_order())

        edges_lines = []
        for e in dag_task.edges:
            desc = e.scenario_description[:200] if e.scenario_description else "data handoff"
            edges_lines.append(f"- **{e.source_skill} -> {e.target_skill}**: {desc}")
        edges_text = "\n".join(edges_lines)

        return f"""
## DAG Workflow Constraint

The task follows a **{dag_task.structure_type}** structure with the following data flow:

**Execution Order** (topological): {topo_order}

**Data Handoffs**:
{edges_text}

**Important**: The execution MUST follow the topological order above. Each upstream skill's output serves as input to its downstream skill(s). Do NOT skip skills or reorder operations.
"""

    def _extract_guiding_metadata(self, response: str) -> str:
        """
        从 LLM 响应中提取引导思路部分。

        Args:
            response: LLM 完整响应

        Returns:
            提取的引导思路
        """
        import re

        # 尝试匹配 "## Guiding Metadata" 后的内容
        match = re.search(
            r'##\s*Guiding\s*Metadata\s*\n(.*)',
            response,
            re.DOTALL | re.IGNORECASE
        )

        if match:
            return match.group(1).strip()

        # Fallback: 返回整个响应
        return response.strip()

    def filter_task_instruction(
        self,
        task_instruction: str,
        max_retries: int = 3,
        dag_task: "DAGTask | None" = None,
    ) -> dict:
        """
        Filter task instruction using quality scoring prompt.

        Evaluates the instruction on 6 dimensions and returns scores.
        When dag_task is provided, also evaluates DAG compliance.
        Raises ValueError if JSON parsing fails after max_retries.

        Args:
            task_instruction: The task instruction to evaluate
            max_retries: Maximum attempts for JSON parsing
            dag_task: Optional DAGTask for DAG compliance checking (V2)

        Returns:
            Dict with dimension scores, weighted_average, and optional dag_compliance
        """
        from ..prompts.filter import instruction_filter

        # Build prompt with instruction
        user_prompt = instruction_filter.replace(
            "[INSERT TASK INSTRUCTION HERE]",
            task_instruction
        )

        result = None
        for attempt in range(max_retries):
            try:
                response = self.llm.generate(
                    system_prompt="You are an expert evaluator. Return only valid JSON.",
                    user_prompt=user_prompt,
                    temperature=0.3,
                )

                # Parse JSON with fallback methods
                result = self._parse_filter_json(response)
                break

            except (json.JSONDecodeError, ValueError) as e:
                if attempt < max_retries - 1:
                    print(f"[Filter] JSON parse attempt {attempt + 1} failed: {e}, retrying...")
                    continue
                raise ValueError(f"Failed to parse filter response after {max_retries} attempts: {e}")

        if result is None:
            raise ValueError("Filter scoring failed")

        # DAG compliance check (V2 only)
        if dag_task is not None:
            dag_compliance = self._evaluate_dag_compliance(
                task_instruction=task_instruction,
                dag_task=dag_task,
                max_retries=max_retries,
            )
            result["dag_compliance"] = dag_compliance

        return result

    def _evaluate_dag_compliance(
        self,
        task_instruction: str,
        dag_task: "DAGTask",
        max_retries: int = 3,
    ) -> dict:
        """
        Evaluate DAG compliance of a task instruction.

        Args:
            task_instruction: The task instruction to evaluate
            dag_task: DAGTask with DAG structure info
            max_retries: Maximum attempts for JSON parsing

        Returns:
            Dict with dag compliance scores (skill_coverage, topological_consistency, edge_semantics, dag_weighted_average)
        """
        from ..prompts.filter import dag_compliance_filter

        # Format DAG structure info
        nodes_list = "\n".join(
            f"- {s['skill_name']}" for s in dag_task.skills
        )
        edges_lines = []
        for e in dag_task.edges:
            desc = f" — {e.scenario_description[:150]}" if e.scenario_description else ""
            edges_lines.append(f"- {e.source_skill} -> {e.target_skill}{desc}")
        edges_list = "\n".join(edges_lines)
        topological_order = " -> ".join(dag_task.topological_order())

        user_prompt = dag_compliance_filter.format(
            structure_type=dag_task.structure_type,
            nodes_list=nodes_list,
            edges_list=edges_list,
            topological_order=topological_order,
            task_instruction=task_instruction,
        )

        for attempt in range(max_retries):
            try:
                response = self.llm.generate(
                    system_prompt="You are an expert evaluator of DAG-constrained tasks. Return only valid JSON.",
                    user_prompt=user_prompt,
                    temperature=0.3,
                )
                return self._parse_filter_json(response)

            except (json.JSONDecodeError, ValueError) as e:
                if attempt < max_retries - 1:
                    print(f"[DAG Filter] JSON parse attempt {attempt + 1} failed: {e}, retrying...")
                    continue
                print(f"[DAG Filter] Failed after {max_retries} attempts: {e}")
                return {
                    "skill_coverage": {"score": 5, "reason": "evaluation failed"},
                    "topological_consistency": {"score": 5, "reason": "evaluation failed"},
                    "edge_semantics": {"score": 5, "reason": "evaluation failed"},
                    "dag_weighted_average": {"score": 5.0, "formula": "fallback"},
                }

        return {"dag_weighted_average": {"score": 5.0, "formula": "fallback"}}

    def _parse_filter_json(self, response: str) -> dict:
        """
        Parse JSON from LLM response with multiple strategies.

        Args:
            response: LLM response text

        Returns:
            Parsed JSON as dict

        Raises:
            ValueError: If JSON cannot be extracted
        """
        import re

        # Strategy 1: Direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from markdown code block
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find JSON object with nested braces
        # Match outermost braces containing nested objects
        brace_pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
        json_match = re.search(brace_pattern, response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not extract JSON from response: {response[:500]}")

    def generate_task_from_dag(
        self,
        dag_task: "DAGTask",
        exploration_summary: str,
        file_summaries: "FileSummaryResult | None" = None,
    ) -> str:
        """
        Generate a task instruction constrained by DAG structure.

        Args:
            dag_task: DAGTask describing the skill dependency graph
            exploration_summary: Exploration summary content
            file_summaries: FileSummaryResult object (optional)

        Returns:
            Generated task instruction
        """
        if self.llm is None:
            raise ValueError("LLM client is required for DAG task generation")

        from ..prompts.dag_task_synthesize import (
            DAG_TASK_SYS_PROMPT,
            DAG_TASK_USER_PROMPT,
            MULTI_FILE_INSTRUCTION,
            select_random_examples,
        )

        # Build nodes list
        nodes_list = "\n".join(
            f"- **{s['skill_name']}** (id: {s.get('skill_id', '?')})"
            for s in dag_task.skills
        )

        # Build edges list
        edges_lines = []
        for e in dag_task.edges:
            desc = f" — {e.scenario_description[:150]}" if e.scenario_description else ""
            edges_lines.append(f"- {e.source_skill} -> {e.target_skill}{desc}")
        edges_list = "\n".join(edges_lines)

        # Topological order
        topo = dag_task.topological_order()
        topological_order = " -> ".join(topo)

        # Suggested task fields
        suggested = dag_task.suggested_task or {}
        suggested_title = suggested.get("title", "N/A")
        suggested_input = suggested.get("input", "N/A")
        suggested_output = suggested.get("expected_output", "N/A")
        workflow = suggested.get("workflow", [])
        if isinstance(workflow, list):
            suggested_workflow = "\n".join(
                f"  {i+1}. {step}" if isinstance(step, str) else f"  {i+1}. {step.get('description', str(step))}"
                for i, step in enumerate(workflow)
            )
        else:
            suggested_workflow = str(workflow)

        # Skills info from skill_doc_paths
        skills_info_lines = []
        for skill_name in dag_task.skill_names:
            doc_path = dag_task.skill_doc_paths.get(skill_name, "")
            skills_info_lines.append(f"- **{skill_name}**: {doc_path}")
        skills_info = "\n".join(skills_info_lines)

        # Files info
        files_info = "No files available"
        file_count = 0
        if file_summaries and hasattr(file_summaries, 'files'):
            file_count = len(file_summaries.files)
            files_lines = []
            for entry in file_summaries.files:
                files_lines.append(f"- **{entry.name}** ({entry.content_type})")
                files_lines.append(f"  Path: {entry.path}")
                files_lines.append(f"  Summary: {entry.summary}\n")
            files_info = "\n".join(files_lines)

        example_tasks = select_random_examples(n=3)

        prompt = DAG_TASK_USER_PROMPT.format(
            structure_type=dag_task.structure_type,
            nodes_list=nodes_list,
            edges_list=edges_list,
            topological_order=topological_order,
            suggested_title=suggested_title,
            suggested_input=suggested_input,
            suggested_workflow=suggested_workflow,
            suggested_output=suggested_output,
            exploration_summary=exploration_summary,
            skills_info=skills_info,
            files_info=files_info,
            example_tasks=example_tasks,
        )

        if file_count > 1:
            prompt += "\n" + MULTI_FILE_INSTRUCTION.format(file_count=file_count)

        print("=" * 80)
        print(f"[InstructionGenerator] Generating DAG-constrained task ({dag_task.structure_type})")
        print("=" * 50)

        response = self.llm.generate(
            system_prompt=DAG_TASK_SYS_PROMPT,
            user_prompt=prompt,
            temperature=0.6,
        )

        return self._extract_task_instruction(response)
