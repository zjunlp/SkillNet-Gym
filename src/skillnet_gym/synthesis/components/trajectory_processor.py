"""Trajectory processor for validation and information extraction"""

import re
from typing import Any

from ..config import (
    FileOperation,
    FileOperationType,
    ProcessedTrajectory,
    Trajectory,
    ValidationResult,
)
from ..execution.trajectory_recorder import TrajectoryRecorder


class TrajectoryProcessor:
    """Processes and validates trajectories for task synthesis"""

    def __init__(self, require_output: bool = True, min_steps: int = 1):
        """
        Initialize processor.

        Args:
            require_output: Whether to require output files for validity
            min_steps: Minimum number of steps for validity
        """
        self.require_output = require_output
        self.min_steps = min_steps
        self.recorder = TrajectoryRecorder()

    def process(self, trajectory: Trajectory) -> ProcessedTrajectory:
        """
        Process a trajectory to extract structured information.

        Args:
            trajectory: Raw trajectory to process

        Returns:
            ProcessedTrajectory with extracted information
        """
        # Extract file operations
        file_operations = self.recorder.extract_file_operations(trajectory)

        # Identify used skills
        used_skills = self.recorder.get_used_skills(trajectory)

        # Extract pip packages from Bash commands
        pip_packages = self._extract_pip_packages(trajectory)

        # Generate summary
        summary = self._generate_summary(trajectory, file_operations)

        # Build metadata
        metadata = {
            "num_steps": trajectory.num_steps,
            "num_tool_uses": len(trajectory.tool_use_steps),
            "num_write_ops": len(trajectory.write_operations),
            "tool_usage": self.recorder.get_tool_usage_summary(trajectory),
            "file_types": list({op.extension for op in file_operations if op.extension}),
        }

        return ProcessedTrajectory(
            trajectory=trajectory,
            file_operations=file_operations,
            used_skills=used_skills,
            pip_packages=pip_packages,
            summary=summary,
            metadata=metadata,
        )

    def validate(self, trajectory: Trajectory) -> ValidationResult:
        """
        Validate a trajectory for task synthesis.

        Args:
            trajectory: Trajectory to validate

        Returns:
            ValidationResult with validity status and any errors/warnings
        """
        result = ValidationResult(is_valid=True)

        # Check execution success
        if not trajectory.success:
            result.add_warning("Trajectory execution was not successful")

        # Check for minimum steps
        if trajectory.num_steps < self.min_steps:
            result.add_error(f"Trajectory has {trajectory.num_steps} steps, minimum is {self.min_steps}")

        # Check for output files if required
        if self.require_output and not trajectory.output_files:
            # Also check write operations
            if not trajectory.write_operations:
                result.add_error("No output files or write operations found")

        # Check for at least one tool use
        if not trajectory.tool_use_steps:
            result.add_error("No tool use operations found")

        # Check for empty trajectory
        if not trajectory.steps:
            result.add_error("Trajectory is empty")

        # Warn about very short trajectories
        if 0 < trajectory.num_steps <= 2:
            result.add_warning("Trajectory has very few steps")

        # Check for errors in trajectory (look for error patterns)
        for step in trajectory.steps:
            if step.action_type == "text" and step.reasoning:
                if any(err in step.reasoning.lower() for err in ["error:", "exception:", "failed to"]):
                    result.add_warning(f"Potential error detected in step {step.step_id}")

        return result

    def _extract_pip_packages(self, trajectory: Trajectory) -> list[str]:
        """Extract pip package names from Bash commands"""
        packages = set()

        for step in trajectory.steps:
            if step.action_type != "tool_use" or step.tool_name != "Bash":
                continue

            if not step.tool_input:
                continue

            command = step.tool_input.get("command", "")

            # Match pip install commands
            # Patterns: pip install X, pip3 install X, python -m pip install X
            patterns = [
                r'pip3?\s+install\s+(?:--[^\s]+\s+)*([^\s&|;]+)',
                r'python3?\s+-m\s+pip\s+install\s+(?:--[^\s]+\s+)*([^\s&|;]+)',
            ]

            for pattern in patterns:
                matches = re.findall(pattern, command)
                for match in matches:
                    # Skip flags and options
                    if not match.startswith("-"):
                        # Clean version specifiers
                        pkg_name = re.split(r'[=<>!]', match)[0]
                        if pkg_name:
                            packages.add(pkg_name)

        return list(packages)

    def _generate_summary(
        self,
        trajectory: Trajectory,
        file_operations: list[FileOperation],
    ) -> str:
        """Generate a human-readable summary of the trajectory"""
        parts = []

        # Basic info
        parts.append(f"Session: {trajectory.session_id}")
        parts.append(f"Steps: {trajectory.num_steps}")
        parts.append(f"Success: {trajectory.success}")

        # Tool usage
        tool_usage = self.recorder.get_tool_usage_summary(trajectory)
        if tool_usage:
            usage_str = ", ".join(f"{k}:{v}" for k, v in sorted(tool_usage.items()))
            parts.append(f"Tools used: {usage_str}")

        # File operations
        creates = [op for op in file_operations if op.operation_type == FileOperationType.CREATE]
        if creates:
            parts.append(f"Files created: {len(creates)}")
            for op in creates[:5]:  # Limit to first 5
                parts.append(f"  - {op.filename}")
            if len(creates) > 5:
                parts.append(f"  ... and {len(creates) - 5} more")

        # Input files
        if trajectory.input_files:
            parts.append(f"Input files: {len(trajectory.input_files)}")

        return "\n".join(parts)

    def summarize_for_llm(self, trajectory: Trajectory) -> str:
        """
        Generate a detailed summary suitable for LLM processing.

        This produces a structured summary that can be used by the
        instruction generator to create task instructions.

        Args:
            trajectory: Trajectory to summarize

        Returns:
            Detailed summary string
        """
        sections = []

        # Overview
        sections.append("## Trajectory Overview")
        sections.append(f"- Total steps: {trajectory.num_steps}")
        sections.append(f"- Execution status: {'Success' if trajectory.success else 'Failed'}")
        sections.append(f"- Duration: {trajectory.duration_ms}ms")
        sections.append("")

        # Input files
        if trajectory.input_files:
            sections.append("## Input Files")
            for f in trajectory.input_files:
                sections.append(f"- {f}")
            sections.append("")

        # Output files
        if trajectory.output_files:
            sections.append("## Output Files")
            for f in trajectory.output_files:
                sections.append(f"- {f}")
            sections.append("")

        # Key operations
        sections.append("## Key Operations")
        for step in trajectory.steps:
            if step.action_type == "tool_use" and step.tool_name:
                if step.tool_name == "Write" and step.tool_input:
                    path = step.tool_input.get("file_path", "unknown")
                    sections.append(f"- Created file: {path}")
                elif step.tool_name == "Read" and step.tool_input:
                    path = step.tool_input.get("file_path", "unknown")
                    sections.append(f"- Read file: {path}")
                elif step.tool_name == "Bash" and step.tool_input:
                    cmd = step.tool_input.get("command", "")
                    if len(cmd) > 80:
                        cmd = cmd[:77] + "..."
                    sections.append(f"- Executed: {cmd}")
        sections.append("")

        # Reasoning excerpts (if available)
        reasoning_steps = [s for s in trajectory.steps if s.reasoning and s.action_type == "text"]
        if reasoning_steps:
            sections.append("## Agent Reasoning (excerpts)")
            for step in reasoning_steps[:3]:  # Limit excerpts
                text = step.reasoning[:500] if len(step.reasoning) > 500 else step.reasoning
                sections.append(f"- {text}")
            sections.append("")

        return "\n".join(sections)

    def extract_output_structure(self, trajectory: Trajectory) -> dict[str, Any]:
        """
        Extract the structure of output files.

        Useful for generating tests that validate output format.

        Args:
            trajectory: Trajectory to analyze

        Returns:
            Dict describing output file structures
        """
        contents = self.recorder.get_output_file_contents(trajectory)
        structures = {}

        for path, content in contents.items():
            ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""

            structure = {
                "path": path,
                "extension": ext,
                "content_length": len(content),
            }

            # Analyze content based on type
            if ext == "json":
                import json
                try:
                    data = json.loads(content)
                    structure["keys"] = list(data.keys()) if isinstance(data, dict) else None
                    structure["is_array"] = isinstance(data, list)
                    structure["valid_json"] = True
                except json.JSONDecodeError:
                    structure["valid_json"] = False

            elif ext == "csv":
                lines = content.strip().split("\n")
                if lines:
                    structure["header"] = lines[0].split(",") if lines else []
                    structure["row_count"] = len(lines) - 1

            elif ext in ("md", "txt"):
                lines = content.strip().split("\n")
                structure["line_count"] = len(lines)
                # Extract headers for markdown
                if ext == "md":
                    headers = [l for l in lines if l.startswith("#")]
                    structure["headers"] = headers[:5]

            structures[path] = structure

        return structures
