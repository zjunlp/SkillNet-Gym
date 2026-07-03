"""Trajectory validator using Process Reward Model (PRM).

This module validates oracle execution trajectories by calling Claude Code
as a process reward model. The PRM can use available skills to verify
that the trajectory is correct.
"""

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..execution.claude_executor import ClaudeExecutor

from ..config import Trajectory, TrajectoryStep
from ..prompts.prm_validation_prompts import (
    PRM_VALIDATION_PROMPT,
    PRM_VALIDATION_SYSTEM,
    PRM_RETRY_PROMPT_TEMPLATE,
    RESPONSE_SUMMARIZATION_PROMPT,
)
from ..utils.llm_client import LLMClient


class PRMValidationResult:
    """Result of PRM validation."""

    def __init__(
        self,
        is_valid: bool,
        issues: list[str] | None = None,
        feedback: str = "",
    ):
        self.is_valid = is_valid
        self.issues = issues or []
        self.feedback = feedback

    def __repr__(self) -> str:
        return f"PRMValidationResult(is_valid={self.is_valid}, issues={len(self.issues)})"


class TrajectoryValidator:
    """Validates oracle trajectories using Process Reward Model (PRM).

    The PRM is implemented by calling Claude Code with the trajectory
    and asking it to verify correctness using available skills.
    """

    def __init__(
        self,
        executor: "ClaudeExecutor",
        llm_client: LLMClient | None = None,
        max_response_chars: int = 15000,
    ):
        """Initialize the validator.

        Args:
            executor: Claude Code executor for PRM validation
            llm_client: LLM client for response summarization (optional)
            max_response_chars: Max characters for tool_response before summarization
        """
        self.executor = executor
        self.llm = llm_client
        self.max_response_chars = max_response_chars

    def extract_trajectory_for_prm(
        self,
        trajectory: Trajectory,
        max_steps: int = 100,
        max_response_chars: int | None = None,
    ) -> str:
        """Extract and format trajectory for PRM validation.

        保留全部类型的 action（包括 tool_use, text），对每轮 tool_response 做截断处理。
        截断策略：保留前 70% + 后 30%，中间标注截断字符数。

        Args:
            trajectory: The trajectory to format
            max_steps: Maximum number of steps to include
            max_response_chars: Max chars for tool_response (uses self.max_response_chars if None)

        Returns:
            Formatted trajectory string (trajectory summary)
        """
        if max_response_chars is None:
            max_response_chars = self.max_response_chars

        formatted_steps = []

        steps_to_process = trajectory.steps
        if len(steps_to_process) > max_steps:
            # Include first half and last half
            half = max_steps // 2
            steps_to_process = (
                trajectory.steps[:half]
                + [TrajectoryStep(step_id=-1, action_type="note", reasoning=f"... ({len(trajectory.steps) - max_steps} steps omitted) ...")]
                + trajectory.steps[-half:]
            )

        for step in steps_to_process:
            # 处理 note 类型（步骤省略标记）
            if step.action_type == "note":
                formatted_steps.append({
                    "note": step.reasoning,
                })
                continue

            # 保留所有类型的 action（不只是 tool_use）
            step_data = {
                "step": step.step_id,
                "action_type": step.action_type,
            }

            # 对于 tool_use 类型，添加工具信息
            if step.action_type == "tool_use":
                step_data["tool"] = step.tool_name
                step_data["action"] = step.tool_input  # 保留完整 action

                # 对 tool_response 做截断（保留前 70% + 后 30%）
                if step.tool_output:
                    output_len = len(step.tool_output)
                    if output_len > max_response_chars:
                        # 保留前 70% 和后 30%
                        head_len = int(max_response_chars * 0.7)
                        tail_len = max_response_chars - head_len
                        truncated_chars = output_len - max_response_chars
                        step_data["response"] = (
                            step.tool_output[:head_len] +
                            f"\n... [{truncated_chars} chars truncated] ...\n" +
                            step.tool_output[-tail_len:]
                        )
                        step_data["response_truncated"] = True
                        step_data["response_original_chars"] = output_len
                    else:
                        step_data["response"] = step.tool_output

            # 对于 text 类型，添加 reasoning
            elif step.action_type == "text":
                if step.reasoning:
                    step_data["reasoning"] = step.reasoning

            formatted_steps.append(step_data)

        return json.dumps(formatted_steps, indent=2, ensure_ascii=False)

    def _summarize_response(self, response: str) -> str:
        """Summarize a large tool response.

        Args:
            response: The full response text

        Returns:
            Summarized response
        """
        if self.llm:
            try:
                prompt = RESPONSE_SUMMARIZATION_PROMPT.format(
                    response=response[:20000]  # Cap input to LLM
                )
                return self.llm.generate(
                    system_prompt="Summarize concisely, preserving key information.",
                    user_prompt=prompt,
                    temperature=0.3,
                )
            except Exception:
                pass

        # Fallback: truncate with markers
        preview_len = self.max_response_chars // 2
        return (
            f"{response[:preview_len]}\n"
            f"[... {len(response) - self.max_response_chars} chars truncated ...]\n"
            f"{response[-preview_len:]}"
        )

    def validate(
        self,
        trajectory: Trajectory,
        task_instruction: str,
        skills_dir: str,
        working_dir: str,
    ) -> PRMValidationResult:
        """Validate trajectory using Claude Code as PRM.

        Args:
            trajectory: The oracle trajectory to validate
            task_instruction: The original task instruction
            skills_dir: Path to skills directory (for PRM to use)
            working_dir: Working directory for PRM execution

        Returns:
            PRMValidationResult with validation status and feedback
        """
        # Extract trajectory for prompt
        trajectory_str = self.extract_trajectory_for_prm(trajectory)

        # Build validation prompt
        prompt = f"{PRM_VALIDATION_SYSTEM}\n\n{PRM_VALIDATION_PROMPT.format(task_instruction=task_instruction, trajectory=trajectory_str, skills_hint=skills_dir)}"

        # Execute PRM validation
        result = self.executor.execute(
            prompt=prompt,
            working_dir=working_dir,
            skills_dir=skills_dir,
        )

        # Parse result
        return self._parse_prm_result(result.output if result.output else "")

    def _parse_prm_result(self, output: str) -> PRMValidationResult:
        """Parse the PRM's output to extract validation result.

        Args:
            output: Raw output from PRM execution

        Returns:
            PRMValidationResult
        """
        # Try to find JSON block in output
        json_patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'\{[^{}]*"is_valid"[^{}]*\}',
        ]

        for pattern in json_patterns:
            match = re.search(pattern, output, re.DOTALL)
            if match:
                try:
                    json_str = match.group(1) if '```' in pattern else match.group(0)
                    data = json.loads(json_str)

                    return PRMValidationResult(
                        is_valid=data.get("is_valid", False),
                        issues=data.get("issues", []),
                        feedback=data.get("feedback", ""),
                    )
                except json.JSONDecodeError:
                    continue

        # Fallback: try to infer from text
        output_lower = output.lower()

        # Check for explicit validation signals
        if "is_valid\": true" in output_lower or "is_valid: true" in output_lower:
            return PRMValidationResult(is_valid=True)

        if "is_valid\": false" in output_lower or "is_valid: false" in output_lower:
            # Extract issues from text
            issues = []
            for line in output.split("\n"):
                if line.strip().startswith("- ") or line.strip().startswith("* "):
                    issues.append(line.strip()[2:])
                elif "issue" in line.lower() or "error" in line.lower():
                    issues.append(line.strip())

            return PRMValidationResult(
                is_valid=False,
                issues=issues[:10],  # Cap at 10 issues
                feedback=output[:2000],  # Use output as feedback
            )

        # If we can't parse, assume valid (conservative)
        return PRMValidationResult(
            is_valid=True,
            feedback="Could not parse PRM result, assuming valid.",
        )

    def build_retry_prompt(
        self,
        original_prompt: str,
        validation_result: PRMValidationResult,
        attempt_number: int,
    ) -> str:
        """Build a retry prompt incorporating PRM feedback.

        Args:
            original_prompt: The original oracle execution prompt
            validation_result: The failed validation result
            attempt_number: Current retry attempt number

        Returns:
            New prompt for oracle retry
        """
        issues_list = "\n".join(
            f"- {issue}" for issue in validation_result.issues
        ) if validation_result.issues else "- No specific issues identified"

        return PRM_RETRY_PROMPT_TEMPLATE.format(
            issues_list=issues_list,
            feedback=validation_result.feedback,
            attempt_number=attempt_number,
            original_prompt=original_prompt,
        )
