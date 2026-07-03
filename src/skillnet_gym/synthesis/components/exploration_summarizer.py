"""Exploration summarizer for extracting summaries from trajectories (programmatic extraction)"""

from ..config import LoopSummary, ExplorationSummaryTable, Trajectory
from ..execution.trajectory_recorder import TrajectoryRecorder
from ..utils.llm_client import LLMClient


# Goal generation prompts
# GOAL_SYSTEM_PROMPT = """You are a task design expert. Based on skill exploration results,
# generate a specific, achievable task for an AI agent to execute.

# You must generate TWO separate sections:
# 1. **Task Instruction** - A clear, concise task description (like examples below). This will be shown to the agent.
# 2. **Execution Guide** - Detailed step-by-step guidance and hints to ensure 100% correct execution. This is for internal use.

# The task instruction should:
# - Be specific and actionable
# - Produce structured output that is verifiable and whose uniqueness can be validated by test code
# - NOT include implementation steps or code
# - Specify output format requirements
# - Styles of task instructions that can be uniquely verified include, but are not limited to, numerical computation, code execution, and deterministic information extraction
# """

# GOAL_GENERATION_PROMPT = """## Good Task Instruction Examples

# Example 1:
# Recover missing values in an Excel file `nasa_budget_incomplete.xlsx`. Missing values are marked with "???". Analyze the relationships between sheets to determine the correct values and replace each "???" with the computed numeric value. Save as `nasa_budget_recovered.xlsx`

# Example 2:
# Calculate the mass of a 3D printed part. The input (`/root/scan_data.stl`) is a binary STL with Material ID stored in the attribute byte count field.

# Parse the binary STL, identify the **largest connected component** (filtering out scanning debris), extract the Material ID, reference `/root/material_density_table.md` for density, and calculate mass using `Volume * Density`.

# Save to `/root/mass_report.json`:
# ```json
# {{
#   "main_part_mass": 12345.67,
#   "material_id": 42
# }}
# ```
# NOTE: Result accuracy within 0.1% is acceptable.

# Example 3:
# You are helping a research team verify the integrity of their bibliography before submitting a paper. The team suspects that some citations in their BibTeX file may be fake or hallucinated.

# The BibTeX file is located at `/root/test.bib` and contains a mix of real and potentially fake academic citations. Your task is to identify which citations are fake or hallucinated.

# Write your findings to `/root/answer.json` in the following format:

# ```json
# {{
#   "fake_citations": [
#     "Title of first fake paper",
#     "Title of second fake paper",
#     ...
#   ]
# }}
# ```

# Notes:
# - Return only the **titles** of the fake citations in the `fake_citations` array
# - Titles should be cleaned (remove BibTeX formatting like `{{}}` or `\`)
# - Sort titles alphabetically for consistency

# ---

# ## Exploration Summary
# {summary_table}

# ## Available Skills
# {skills}

# ## Available Files
# {files}

# ---

# Generate the task in the following format (use these exact headers):

# ## Task Instruction
# [Clear, concise task description. DO NOT include step-by-step implementation details.]

# ## Execution Guide
# [Detailed guidance for execution, including:
# - Key insights discovered during exploration
# - Recommended approach and tools to use
# - Potential pitfalls and how to avoid them
# - Expected intermediate results
# - Verification criteria]
# """


GOAL_GENERATION_PROMPT = """You must generate TWO separate sections:
- **Task Instruction** - A clear, concise task description (like examples below). This will be shown to the agent.
- **Execution Guide** - Detailed step-by-step guidance and hints to ensure 100% correct execution. This is for internal use.

## Exploration Summary
<Exploration Summary>
{summary_table}
</Exploration Summary>

## Available Skills
{skills}

## Available Files
{files}

---

## Task Instruction Synthesized Guidelines

1. Understand the logical relationships between workflows and functions in the exploration summary; this will help you generate complex yet solvable task instructions.

2. Do not mention any metadata related to the Execution Guide in the task instructions.

3. The Task Instruction produce structured output must be verifiable and whose uniqueness can be validated by test code

4. The files mentioned as input may only be those listed under 'Available Files'

5. Task instructions do not need to include all of the functions mentioned in the exploration summary

6. Styles of task instructions that can be uniquely verified include, but are not limited to, numerical computation, code execution, and deterministic information extraction

---

## Good Task Instruction Examples

Example 1:
Recover missing values in an Excel file `nasa_budget_incomplete.xlsx`. Missing values are marked with "???". Analyze the relationships between sheets to determine the correct values and replace each "???" with the computed numeric value. Save as `nasa_budget_recovered.xlsx`

Example 2:
Calculate the mass of a 3D printed part. The input (`/root/scan_data.stl`) is a binary STL with Material ID stored in the attribute byte count field.

Parse the binary STL, identify the **largest connected component** (filtering out scanning debris), extract the Material ID, reference `/root/material_density_table.md` for density, and calculate mass using `Volume * Density`.

Save to `/root/mass_report.json`:
```json
{{
  "main_part_mass": 12345.67,
  "material_id": 42
}}
```
NOTE: Result accuracy within 0.1% is acceptable.

Example 3:
Under the `/workspace/` folder, there is a legacy user management microservice that you need to migrate from Java 8/Spring Boot 2.7 to Java 21/Spring Boot 3.2. This microservice is based on REST API and manages user information. Key functions are: CRUD operations for user management, JWT-based authentication, Role-based access control, Database persistence with JPA/Hibernate, Input validation, and External API calls using RestTemplate.

You migration should upgrade Java version from 8 to 21, Spring Boot version from 2.7.x to 3.2.x and upgrade all related dependencies to compatible versions. Make sure the namespace migration are properly handled. You also need to ensure Hibernate 6 compatibility, update to Spring Security 6, migrate RestTemplate to RestClient, and fix deprecated security methods.

Make sure to run the following commands to check the quality:

1. `mvn clean compile` (you code should not return errors)
2. `mvn test` (your code need to pass all unit tests)
 
---

## Execution Guide Synthesized Guidelines

1. Understand and identify the workflow patterns discovered in the exploration summary.

2. Recommend the processing methods and tools mentioned in the exploration summary; do not generate methods that do not appear in the exploration summary.

3. Generate possible pitfalls and how to avoid them.

4. Specify the expected intermediate results.

5. Include an agent self-verification loop mechanism.

---

## Output

Generate the task in the following format (use these exact headers):

## Task Instruction
[Clear, concise task description.]

## Execution Guide
[Detailed guidance for execution]
"""

class ExplorationSummarizer:
    """从探索轨迹中程序化提取总结信息"""

    def __init__(self, llm_client: LLMClient | None = None):
        """
        Initialize the exploration summarizer.

        Args:
            llm_client: LLM client for Goal generation (optional, only needed for generate_goal)
        """
        self.llm = llm_client  # 仅用于 Goal 生成
        self.recorder = TrajectoryRecorder()

    def summarize_trajectory(self, trajectory: Trajectory, loop_index: int) -> LoopSummary:
        """
        程序化提取单次循环的总结。

        Args:
            trajectory: The trajectory to summarize
            loop_index: The index of this exploration loop (0-based)

        Returns:
            LoopSummary with extracted information
        """
        # 从 trajectory 中提取信息（复用已有方法）
        used_skills = self.recorder.get_used_skills(trajectory)
        tool_usage = self.recorder.get_tool_usage_summary(trajectory)
        bash_commands = self.recorder.get_bash_commands(trajectory)

        # 从 steps 中提取读写文件
        read_files = []
        written_files = []
        edited_files = []

        for step in trajectory.steps:
            if step.action_type != "tool_use" or not step.tool_input:
                continue
            if step.tool_name == "Read":
                path = step.tool_input.get("file_path", "")
                if path and path not in read_files:
                    read_files.append(path)
            elif step.tool_name == "Write":
                path = step.tool_input.get("file_path", "")
                if path and path not in written_files:
                    written_files.append(path)
            elif step.tool_name == "Edit":
                path = step.tool_input.get("file_path", "")
                if path and path not in edited_files:
                    edited_files.append(path)

        return LoopSummary(
            loop_index=loop_index,
            trajectory=trajectory,
            used_skills=used_skills,
            bash_commands=bash_commands,
            read_files=read_files,
            written_files=written_files,
            edited_files=edited_files,
            tool_usage=tool_usage,
        )

    def generate_summary_table(self, loop_summaries: list[LoopSummary]) -> ExplorationSummaryTable:
        """
        合并所有循环的总结为总结表。

        Args:
            loop_summaries: List of LoopSummary objects from each exploration loop

        Returns:
            ExplorationSummaryTable with aggregated information
        """
        all_skills = []
        all_commands = []
        all_read = []
        all_written = []
        total_usage: dict[str, int] = {}

        for summary in loop_summaries:
            all_skills.extend(summary.used_skills)
            all_commands.extend(summary.bash_commands)
            all_read.extend(summary.read_files)
            all_written.extend(summary.written_files)
            for tool, count in summary.tool_usage.items():
                total_usage[tool] = total_usage.get(tool, 0) + count

        return ExplorationSummaryTable(
            loop_summaries=loop_summaries,
            total_skills_used=list(dict.fromkeys(all_skills)),
            all_bash_commands=list(dict.fromkeys(all_commands)),
            all_read_files=list(dict.fromkeys(all_read)),
            all_written_files=list(dict.fromkeys(all_written)),
            total_tool_usage=total_usage,
        )

    def format_summary_for_prompt(self, summary_table: ExplorationSummaryTable) -> str:
        """
        格式化总结表为可读文本（用于下一次 prompt）。

        Args:
            summary_table: The summary table to format

        Returns:
            Formatted string representation of the summary
        """
        lines = []
        lines.append("## Exploration Summary")
        lines.append(f"- Total loops: {len(summary_table.loop_summaries)}")
        lines.append(f"- Skills used: {', '.join(summary_table.total_skills_used) or 'None'}")
        lines.append(f"- Files read: {len(summary_table.all_read_files)}")
        lines.append(f"- Files written: {len(summary_table.all_written_files)}")
        lines.append("")
        lines.append("### Tool Usage")
        for tool, count in sorted(summary_table.total_tool_usage.items()):
            lines.append(f"- {tool}: {count} times")
        lines.append("")
        lines.append("### Commands Executed")
        for cmd in summary_table.all_bash_commands[:10]:  # 限制数量
            truncated = cmd[:80] + "..." if len(cmd) > 80 else cmd
            lines.append(f"- {truncated}")
        if len(summary_table.all_bash_commands) > 10:
            lines.append(f"- ... and {len(summary_table.all_bash_commands) - 10} more commands")
        lines.append("")
        lines.append("### Files Written")
        for f in summary_table.all_written_files[:10]:
            lines.append(f"- {f}")
        if len(summary_table.all_written_files) > 10:
            lines.append(f"- ... and {len(summary_table.all_written_files) - 10} more files")
        return "\n".join(lines)

    def generate_goal(
        self,
        exploration_summary: str,
        available_skills: list[str],
        file_summaries: dict[str, str],
    ) -> tuple[str, str]:
        """
        使用 LLM 从探索总结生成任务指令和执行指南。

        Args:
            exploration_summary: 探索总结文件内容（自然语言）
            available_skills: List of available skill names
            file_summaries: Mapping of file paths to their summaries

        Returns:
            Tuple of (task_instruction, execution_guide)
            - task_instruction: 简洁的任务指令（用于最终 HarborTask）
            - execution_guide: 详细的执行指南/元信息（用于 Goal-Driven 执行）

        Raises:
            ValueError: If LLM client is not configured
        """
        if self.llm is None:
            raise ValueError("LLM client required for goal generation")

        skills_text = ", ".join(available_skills) if available_skills else "None specified"
        files_text = "\n".join(f"- {k}: {v}" for k, v in file_summaries.items()) if file_summaries else "No files available"

        prompt = GOAL_GENERATION_PROMPT.format(
            summary_table=exploration_summary,
            skills=skills_text,
            files=files_text,
        )

        response = self.llm.generate(
            system_prompt="You are a task design expert. Based on skill exploration results, generate a specific, achievable task for an AI agent to execute.",
            # GOAL_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.7,
        )

        # 解析响应，提取 Task Instruction 和 Execution Guide
        return self._parse_goal_response(response)

    def _parse_goal_response(self, response: str) -> tuple[str, str]:
        """
        解析 LLM 响应，提取任务指令和执行指南。

        Args:
            response: LLM 生成的完整响应

        Returns:
            Tuple of (task_instruction, execution_guide)
        """
        import re

        # 尝试匹配 ## Task Instruction 和 ## Execution Guide
        task_match = re.search(
            r'##\s*Task\s*Instruction\s*\n(.*?)(?=##\s*Execution\s*Guide|\Z)',
            response,
            re.DOTALL | re.IGNORECASE
        )
        guide_match = re.search(
            r'##\s*Execution\s*Guide\s*\n(.*?)(?=##|\Z)',
            response,
            re.DOTALL | re.IGNORECASE
        )

        task_instruction = task_match.group(1).strip() if task_match else response.strip()
        execution_guide = guide_match.group(1).strip() if guide_match else ""

        # 如果没有找到明确分隔，尝试使用旧格式的 **Goal** 作为 fallback
        if not task_match:
            goal_match = re.search(r'\*\*Goal\*\*:\s*(.*?)(?=\*\*Expected|\*\*Key|\Z)', response, re.DOTALL)
            if goal_match:
                task_instruction = goal_match.group(1).strip()
                # 其余部分作为执行指南
                execution_guide = response[goal_match.end():].strip()

        return task_instruction, execution_guide

    def format_loop_summary(self, loop_summary: LoopSummary) -> str:
        """
        Format a single loop summary for display.

        Args:
            loop_summary: The loop summary to format

        Returns:
            Formatted string representation
        """
        lines = []
        lines.append(f"### Loop {loop_summary.loop_index + 1} Summary")
        lines.append(f"- Steps: {loop_summary.trajectory.num_steps}")
        lines.append(f"- Skills: {', '.join(loop_summary.used_skills) or 'None'}")
        lines.append(f"- Files read: {len(loop_summary.read_files)}")
        lines.append(f"- Files written: {len(loop_summary.written_files)}")
        lines.append(f"- Files edited: {len(loop_summary.edited_files)}")
        lines.append(f"- Bash commands: {len(loop_summary.bash_commands)}")
        return "\n".join(lines)
