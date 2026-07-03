"""Trajectory recorder for extracting structured trajectories from Claude Code output"""

import re
from datetime import datetime
from typing import Any

from ..config import (
    FileOperation,
    FileOperationType,
    ToolCategory,
    TOOL_CATEGORY_MAP,
    Trajectory,
    TrajectoryStep,
)
from .claude_executor import ExecutionResult


# All supported Claude Code tools
SUPPORTED_TOOLS = [
    # File operations
    "Read", "Write", "Edit", "NotebookEdit", "Glob", "Grep",
    # Command execution
    "Bash",
    # Skill invocation
    "Skill",
    # Web operations
    "WebFetch", "WebSearch",
    # Task management
    "Task", "TaskOutput", "TaskStop", "TodoWrite",
    # Plan mode
    "EnterPlanMode", "ExitPlanMode",
    # Worktree
    "EnterWorktree", "ExitWorktree",
    # Cron jobs
    "CronCreate", "CronDelete", "CronList",
    # User interaction
    "AskUserQuestion",
]


class TrajectoryRecorder:
    """Extracts structured trajectories from Claude Code execution results"""

    def record(self, result: ExecutionResult) -> Trajectory:
        """
        Parse execution result into a structured Trajectory.

        Args:
            result: ExecutionResult from ClaudeExecutor

        Returns:
            Trajectory object with parsed steps
        """
        if not result.trajectory_data:
            return self._create_empty_trajectory(result)

        events = result.trajectory_data.get("events", [])
        session_id = result.trajectory_data.get("session_id", "unknown")

        steps = []
        model = ""
        step_id = 0

        # Extended tracking collections
        input_files = []
        output_files = []
        edited_files = []
        notebook_files = []
        used_skills = []
        bash_commands = []
        web_urls = []
        search_queries = []
        glob_patterns = []
        grep_patterns = []

        # Track tool_id -> step_index for filling tool_output later
        pending_tool_outputs: dict[str, int] = {}

        for event in events:
            event_type = event.get("type")

            if event_type == "assistant":
                # Parse assistant message for tool uses and text
                message = event.get("message", {})
                content = message.get("content", [])
                model = model or message.get("model", "")

                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type")

                        if item_type == "tool_use":
                            step = self._parse_tool_use(item, step_id)
                            steps.append(step)

                            # Record tool_id for later output filling
                            tool_id = item.get("id", "")
                            if tool_id:
                                pending_tool_outputs[tool_id] = step_id

                            step_id += 1

                            # Track all tool operations
                            self._track_tool_operation(
                                step=step,
                                input_files=input_files,
                                output_files=output_files,
                                edited_files=edited_files,
                                notebook_files=notebook_files,
                                used_skills=used_skills,
                                bash_commands=bash_commands,
                                web_urls=web_urls,
                                search_queries=search_queries,
                                glob_patterns=glob_patterns,
                                grep_patterns=grep_patterns,
                            )

                        elif item_type == "text":
                            text = item.get("text", "")
                            if text.strip():
                                step = TrajectoryStep(
                                    step_id=step_id,
                                    action_type="text",
                                    reasoning=text,
                                    timestamp=datetime.now().isoformat(),
                                )
                                steps.append(step)
                                step_id += 1

            elif event_type == "user":
                # Parse tool results
                message = event.get("message", {})
                content = message.get("content", [])
                tool_result = event.get("tool_use_result", {})
                tool_use_result = event.get("toolUseResult", {})

                # Extract tool_output from tool_result content items
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        tool_use_id = item.get("tool_use_id", "")
                        result_content = item.get("content", "")

                        # Convert content to string if needed
                        if isinstance(result_content, list):
                            # Handle list of content blocks
                            text_parts = []
                            for block in result_content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    text_parts.append(block.get("text", ""))
                                elif isinstance(block, str):
                                    text_parts.append(block)
                            result_content = "\n".join(text_parts)
                        elif not isinstance(result_content, str):
                            result_content = str(result_content)

                        # Fill in tool_output for the corresponding step
                        if tool_use_id in pending_tool_outputs:
                            step_idx = pending_tool_outputs[tool_use_id]
                            # Truncate output to prevent context overflow
                            truncated_output = self._truncate_output(result_content)
                            steps[step_idx].tool_output = truncated_output
                            del pending_tool_outputs[tool_use_id]

                # Extract file creation info from tool results
                if tool_result and isinstance(tool_result, dict):
                    op_type = tool_result.get("type")
                    file_path = tool_result.get("filePath", "")

                    if op_type == "create" and file_path:
                        if file_path not in output_files:
                            output_files.append(file_path)

                # Extract skills from Glob results (toolUseResult.filenames)
                if tool_use_result and isinstance(tool_use_result, dict):
                    filenames = tool_use_result.get("filenames", [])
                    for filename in filenames:
                        if ".claude/skills/" in filename or "/skills/" in filename:
                            parts = filename.split("/skills/")
                            if len(parts) > 1:
                                skill_part = parts[1].split("/")[0]
                                if skill_part and skill_part not in used_skills:
                                    used_skills.append(skill_part)

            elif event_type == "result":
                # Extract final result info
                pass

        # Deduplicate all collections
        input_files = list(dict.fromkeys(input_files))
        output_files = list(dict.fromkeys(output_files))
        edited_files = list(dict.fromkeys(edited_files))
        notebook_files = list(dict.fromkeys(notebook_files))
        used_skills = list(dict.fromkeys(used_skills))
        bash_commands = list(dict.fromkeys(bash_commands))
        web_urls = list(dict.fromkeys(web_urls))
        search_queries = list(dict.fromkeys(search_queries))
        glob_patterns = list(dict.fromkeys(glob_patterns))
        grep_patterns = list(dict.fromkeys(grep_patterns))

        # Filter out directories from output files (keep only actual files)
        filtered_outputs = []
        for path in output_files:
            filename = path.split("/")[-1]
            if "." in filename:
                filtered_outputs.append(path)
            elif path not in input_files:
                filtered_outputs.append(path)
        output_files = filtered_outputs

        # Also filter input files that are likely outputs (e.g., /root/output/*)
        filtered_inputs = [f for f in input_files if not f.startswith("/root/output")]
        input_files = filtered_inputs

        return Trajectory(
            session_id=session_id,
            steps=steps,
            input_files=input_files,
            output_files=output_files,
            success=result.success,
            duration_ms=result.duration_ms,
            model=model,
            raw_events=events,
            # Extended fields
            used_skills=used_skills,
            bash_commands=bash_commands,
            web_urls=web_urls,
            search_queries=search_queries,
            glob_patterns=glob_patterns,
            grep_patterns=grep_patterns,
            edited_files=edited_files,
            notebook_files=notebook_files,
        )

    def _track_tool_operation(
        self,
        step: TrajectoryStep,
        input_files: list[str],
        output_files: list[str],
        edited_files: list[str],
        notebook_files: list[str],
        used_skills: list[str],
        bash_commands: list[str],
        web_urls: list[str],
        search_queries: list[str],
        glob_patterns: list[str],
        grep_patterns: list[str],
    ) -> None:
        """
        Track tool operation and extract relevant information.

        Handles all Claude Code tools and extracts appropriate data.
        """
        if not step.tool_input:
            return

        tool_name = step.tool_name
        tool_input = step.tool_input

        # === File Write Operations ===
        if tool_name == "Write":
            file_path = tool_input.get("file_path", "")
            if file_path:
                output_files.append(file_path)

        elif tool_name == "Edit":
            file_path = tool_input.get("file_path", "")
            if file_path:
                edited_files.append(file_path)
                # Also add to output_files as it modifies files
                output_files.append(file_path)

        elif tool_name == "NotebookEdit":
            notebook_path = tool_input.get("notebook_path", "")
            if notebook_path:
                notebook_files.append(notebook_path)
                output_files.append(notebook_path)

        # === File Read Operations ===
        elif tool_name == "Read":
            file_path = tool_input.get("file_path", "")
            if file_path:
                input_files.append(file_path)

        elif tool_name == "Glob":
            pattern = tool_input.get("pattern", "")
            if pattern:
                glob_patterns.append(pattern)

        elif tool_name == "Grep":
            pattern = tool_input.get("pattern", "")
            if pattern:
                grep_patterns.append(pattern)

        # === Skill Invocation ===
        elif tool_name == "Skill":
            skill_name = tool_input.get("skill", "")
            if skill_name:
                used_skills.append(skill_name)

        # === Command Execution ===
        elif tool_name == "Bash":
            command = tool_input.get("command", "")
            if command:
                bash_commands.append(command)
                # Also try to extract skills from bash commands
                self._extract_skills_from_bash(command, used_skills)
                # Extract file operations from bash commands
                self._extract_files_from_bash(command, input_files, output_files)

        # === Web Operations ===
        elif tool_name == "WebFetch":
            url = tool_input.get("url", "")
            if url:
                web_urls.append(url)

        elif tool_name == "WebSearch":
            query = tool_input.get("query", "")
            if query:
                search_queries.append(query)

    def _extract_skills_from_bash(self, command: str, used_skills: list[str]) -> None:
        """Extract skill names from bash commands that invoke skill scripts"""
        # Match patterns like: /skills/pdf/..., .claude/skills/excel/...
        matches = re.findall(r'/skills/([^/\s]+)', command)
        for match in matches:
            if match and match not in used_skills:
                used_skills.append(match)

    # System paths that should be excluded from file tracking
    EXCLUDED_PATHS = frozenset([
        '/dev/null',
        '/dev/zero',
        '/dev/random',
        '/dev/urandom',
        '/dev/stdin',
        '/dev/stdout',
        '/dev/stderr',
        '/dev/tty',
    ])

    def _extract_files_from_bash(
        self,
        command: str,
        input_files: list[str],
        output_files: list[str],
    ) -> None:
        """Extract file paths from bash commands"""
        # Common patterns for file operations in bash
        # Output redirection: > file, >> file
        output_matches = re.findall(r'>\s*([^\s&|;>]+)', command)
        for match in output_matches:
            if match and not match.startswith('-') and match not in self.EXCLUDED_PATHS:
                output_files.append(match)

        # Input redirection: < file
        input_matches = re.findall(r'<\s*([^\s&|;<]+)', command)
        for match in input_matches:
            if match and not match.startswith('-') and match not in self.EXCLUDED_PATHS:
                input_files.append(match)

        # Python file write patterns in heredocs
        # Match direct path: open("/path", "w")
        direct_matches = re.findall(r'open\s*\(\s*["\']([^"\']+)["\']\s*,\s*["\']w', command)
        for match in direct_matches:
            if match and match.startswith('/') and match not in self.EXCLUDED_PATHS:
                output_files.append(match)

        # Match variable assignment pattern: out_path = "/path" ... open(out_path, "w")
        # Look for path variables assigned then used with open()
        var_assignments = re.findall(r'(\w+)\s*=\s*["\']([^"\']+\.(?:json|txt|csv|md|py|sh))["\']', command)
        for var_name, path in var_assignments:
            # Check if this variable is used with open(..., "w")
            if re.search(rf'open\s*\(\s*{var_name}\s*,\s*["\']w', command):
                if path.startswith('/') and path not in self.EXCLUDED_PATHS:
                    output_files.append(path)

    def _parse_tool_use(self, item: dict[str, Any], step_id: int) -> TrajectoryStep:
        """Parse a tool_use item into a TrajectoryStep"""
        tool_name = item.get("name", "unknown")
        tool_input = item.get("input", {})
        tool_id = item.get("id", "")

        return TrajectoryStep(
            step_id=step_id,
            action_type="tool_use",
            tool_name=tool_name,
            tool_input=tool_input,
            timestamp=datetime.now().isoformat(),
        )

    def _truncate_output(self, output: str, max_chars: int = 2000) -> str:
        """
        Truncate tool output to prevent context overflow.

        Keeps head (70%) and tail (30%) of the output for better context.

        Args:
            output: Raw output string
            max_chars: Maximum characters to keep

        Returns:
            Truncated output string
        """
        if not output or len(output) <= max_chars:
            return output

        # Keep 70% head, 30% tail
        head_chars = int(max_chars * 0.7)
        tail_chars = int(max_chars * 0.3)

        truncated_len = len(output) - max_chars
        return (
            output[:head_chars] +
            f"\n... [truncated {truncated_len} chars] ...\n" +
            output[-tail_chars:]
        )

    def _create_empty_trajectory(self, result: ExecutionResult) -> Trajectory:
        """Create an empty trajectory when no data is available"""
        return Trajectory(
            session_id="unknown",
            steps=[],
            input_files=[],
            output_files=[],
            success=result.success,
            duration_ms=result.duration_ms,
            model="",
            raw_events=[],
        )

    def extract_file_operations(self, trajectory: Trajectory) -> list[FileOperation]:
        """
        Extract all file operations from a trajectory.

        Args:
            trajectory: Trajectory to analyze

        Returns:
            List of FileOperation objects
        """
        operations = []

        for step in trajectory.steps:
            if step.action_type != "tool_use":
                continue

            if step.tool_name == "Write" and step.tool_input:
                file_path = step.tool_input.get("file_path", "")
                content = step.tool_input.get("content", "")
                if file_path:
                    operations.append(FileOperation(
                        operation_type=FileOperationType.CREATE,
                        file_path=file_path,
                        content=content,
                        timestamp=step.timestamp,
                    ))

            elif step.tool_name == "Edit" and step.tool_input:
                file_path = step.tool_input.get("file_path", "")
                if file_path:
                    operations.append(FileOperation(
                        operation_type=FileOperationType.EDIT,
                        file_path=file_path,
                        timestamp=step.timestamp,
                    ))

            elif step.tool_name == "NotebookEdit" and step.tool_input:
                notebook_path = step.tool_input.get("notebook_path", "")
                new_source = step.tool_input.get("new_source", "")
                if notebook_path:
                    operations.append(FileOperation(
                        operation_type=FileOperationType.NOTEBOOK_EDIT,
                        file_path=notebook_path,
                        content=new_source,
                        timestamp=step.timestamp,
                    ))

            elif step.tool_name == "Read" and step.tool_input:
                file_path = step.tool_input.get("file_path", "")
                if file_path:
                    operations.append(FileOperation(
                        operation_type=FileOperationType.READ,
                        file_path=file_path,
                        timestamp=step.timestamp,
                    ))

        return operations

    def get_created_files(self, trajectory: Trajectory) -> list[str]:
        """
        Get list of files created during execution.

        Args:
            trajectory: Trajectory to analyze

        Returns:
            List of created file paths
        """
        created = []
        for op in self.extract_file_operations(trajectory):
            if op.operation_type == FileOperationType.CREATE:
                created.append(op.file_path)
        return created

    # Skills that are internal/system and should be excluded from task requirements
    INTERNAL_SKILLS = frozenset([
        'update-config',
        'statusline-setup',
        'claude-code-guide',
        'loop',
        'simplify',
    ])

    def get_used_skills(self, trajectory: Trajectory) -> list[str]:
        """
        Get skills used in the trajectory.

        Returns skills from:
        1. Direct Skill tool invocations (recorded during parsing)
        2. Skill paths in file operations (Read/Write/Edit)
        3. Skill patterns in Glob operations
        4. Skill scripts called from Bash commands

        Internal/system skills (update-config, etc.) are excluded.

        Args:
            trajectory: Trajectory to analyze

        Returns:
            List of skill names that were used (excluding internal skills)
        """
        skills = set()

        # First, add skills already tracked during record()
        if trajectory.used_skills:
            skills.update(trajectory.used_skills)

        # Then, also check file paths for additional skill references
        for step in trajectory.steps:
            if step.action_type != "tool_use":
                continue

            # Check for skill paths in Read/Write/Edit operations
            if step.tool_name in ("Read", "Write", "Edit", "NotebookEdit") and step.tool_input:
                file_path = step.tool_input.get("file_path", "") or step.tool_input.get("notebook_path", "")
                self._extract_skill_from_path(file_path, skills)

            # Check for skill paths in Glob patterns
            elif step.tool_name == "Glob" and step.tool_input:
                pattern = step.tool_input.get("pattern", "")
                self._extract_skill_from_path(pattern, skills)

        # Filter out internal/system skills
        skills = skills - self.INTERNAL_SKILLS
        return list(skills)

    def _extract_skill_from_path(self, path: str, skills: set[str]) -> None:
        """Extract skill name from a path containing /skills/ or .claude/skills/"""
        if not path:
            return
        if ".claude/skills/" in path or "/skills/" in path:
            parts = path.split("/skills/")
            if len(parts) > 1:
                # Get the first component after /skills/
                skill_part = parts[1].split("/")[0]
                # Handle glob patterns like **/* by taking the actual directory name
                if skill_part and skill_part not in ("**", "*"):
                    skills.add(skill_part)

    def get_tool_usage_summary(self, trajectory: Trajectory) -> dict[str, int]:
        """
        Get summary of tool usage counts.

        Args:
            trajectory: Trajectory to analyze

        Returns:
            Dict mapping tool names to usage counts
        """
        usage = {}
        for step in trajectory.steps:
            if step.action_type == "tool_use" and step.tool_name:
                usage[step.tool_name] = usage.get(step.tool_name, 0) + 1
        return usage

    def get_output_file_contents(self, trajectory: Trajectory) -> dict[str, str]:
        """
        Extract contents of created files from trajectory.

        Args:
            trajectory: Trajectory to analyze

        Returns:
            Dict mapping file paths to their content
        """
        contents = {}

        for step in trajectory.steps:
            if step.action_type == "tool_use":
                if step.tool_name == "Write" and step.tool_input:
                    file_path = step.tool_input.get("file_path", "")
                    content = step.tool_input.get("content", "")
                    if file_path and content:
                        contents[file_path] = content
                elif step.tool_name == "NotebookEdit" and step.tool_input:
                    notebook_path = step.tool_input.get("notebook_path", "")
                    new_source = step.tool_input.get("new_source", "")
                    if notebook_path and new_source:
                        # Append to existing content for notebooks
                        if notebook_path in contents:
                            contents[notebook_path] += f"\n---\n{new_source}"
                        else:
                            contents[notebook_path] = new_source

        return contents

    def get_bash_commands(self, trajectory: Trajectory) -> list[str]:
        """
        Get all bash commands executed in the trajectory.

        Args:
            trajectory: Trajectory to analyze

        Returns:
            List of bash commands
        """
        if trajectory.bash_commands:
            return trajectory.bash_commands

        commands = []
        for step in trajectory.steps:
            if step.action_type == "tool_use" and step.tool_name == "Bash":
                if step.tool_input:
                    command = step.tool_input.get("command", "")
                    if command:
                        commands.append(command)
        return commands

    def get_web_operations(self, trajectory: Trajectory) -> dict[str, list[str]]:
        """
        Get all web operations (fetches and searches) from trajectory.

        Args:
            trajectory: Trajectory to analyze

        Returns:
            Dict with 'urls' and 'queries' lists
        """
        return {
            "urls": trajectory.web_urls or [],
            "queries": trajectory.search_queries or [],
        }

    def get_search_patterns(self, trajectory: Trajectory) -> dict[str, list[str]]:
        """
        Get all search patterns (glob and grep) from trajectory.

        Args:
            trajectory: Trajectory to analyze

        Returns:
            Dict with 'glob' and 'grep' pattern lists
        """
        return {
            "glob": trajectory.glob_patterns or [],
            "grep": trajectory.grep_patterns or [],
        }

    def get_tool_usage_by_category(self, trajectory: Trajectory) -> dict[str, list[str]]:
        """
        Get tool usage grouped by category.

        Args:
            trajectory: Trajectory to analyze

        Returns:
            Dict mapping category names to lists of tool names used
        """
        usage: dict[str, list[str]] = {}

        for step in trajectory.steps:
            if step.action_type == "tool_use" and step.tool_name:
                category = TOOL_CATEGORY_MAP.get(step.tool_name, ToolCategory.OTHER)
                category_name = category.value
                if category_name not in usage:
                    usage[category_name] = []
                if step.tool_name not in usage[category_name]:
                    usage[category_name].append(step.tool_name)

        return usage

    def get_trajectory_summary(self, trajectory: Trajectory) -> dict[str, Any]:
        """
        Get a comprehensive summary of the trajectory.

        Args:
            trajectory: Trajectory to analyze

        Returns:
            Dict with summary information
        """
        return {
            "session_id": trajectory.session_id,
            "success": trajectory.success,
            "duration_ms": trajectory.duration_ms,
            "total_steps": trajectory.num_steps,
            "tool_use_steps": len(trajectory.tool_use_steps),
            "input_files": trajectory.input_files,
            "output_files": trajectory.output_files,
            "edited_files": trajectory.edited_files,
            "notebook_files": trajectory.notebook_files,
            "used_skills": trajectory.used_skills,
            "bash_commands_count": len(trajectory.bash_commands),
            "web_urls_count": len(trajectory.web_urls),
            "search_queries_count": len(trajectory.search_queries),
            "tool_usage": self.get_tool_usage_summary(trajectory),
            "tool_usage_by_category": self.get_tool_usage_by_category(trajectory),
        }
