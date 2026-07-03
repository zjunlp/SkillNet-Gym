"""Claude Code executor for autonomous task execution.

The Claude CLI reads ``ANTHROPIC_AUTH_TOKEN`` and (optionally)
``ANTHROPIC_BASE_URL`` from the environment. Set them before running the
pipeline, e.g.::

    export ANTHROPIC_AUTH_TOKEN=sk-...
    export ANTHROPIC_BASE_URL=https://api.anthropic.com  # or a proxy
"""
import os

# import json
# import shlex
# import subprocess
# from dataclasses import dataclass, field
# from pathlib import Path
# from typing import Any

# from ..config import ExecutionConfig, PipelineConfig
# from ..utils.file_utils import (
#     copy_directory,
#     copy_file,
#     create_working_directory,
#     ensure_directory,
# )


# @dataclass
# class ExecutionResult:
#     """Result of Claude Code execution"""
#     success: bool
#     output: str
#     exit_code: int
#     trajectory_data: dict[str, Any] | None = None
#     working_dir: str | None = None
#     duration_ms: int = 0
#     error: str | None = None

#     @property
#     def has_trajectory(self) -> bool:
#         return self.trajectory_data is not None and len(self.trajectory_data.get("events", [])) > 0


# class ClaudeExecutor:
#     """Executor for running Claude Code tasks"""

#     def __init__(
#         self,
#         model: str = "claude-opus-4-5-20251124",
#         max_turns: int = 20,
#         timeout: int = 300,
#         api_key: str | None = None,
#         base_url: str | None = None,
#     ):
#         """
#         Initialize Claude executor.

#         Args:
#             model: Model name to use
#             max_turns: Maximum conversation turns
#             timeout: Execution timeout in seconds
#             api_key: API key (defaults to env var)
#             base_url: API base URL (defaults to env var)
#         """
#         self.model = model
#         self.max_turns = max_turns
#         self.timeout = timeout
#         self.api_key = api_key or os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
#         self.base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL")

#     @classmethod
#     def from_config(cls, config: PipelineConfig) -> "ClaudeExecutor":
#         """Create executor from pipeline config"""
#         return cls(
#             model=config.model,
#             max_turns=config.max_turns,
#             timeout=config.timeout,
#         )

#     def execute(
#         self,
#         prompt: str,
#         working_dir: str | None = None,
#         skills_dir: str | None = None,
#         input_file: str | None = None,
#     ) -> ExecutionResult:
#         """
#         Execute a Claude Code task.

#         Args:
#             prompt: The prompt/instruction for Claude
#             working_dir: Working directory (creates temp if None)
#             skills_dir: Skills directory to copy
#             input_file: Input file to copy to working dir

#         Returns:
#             ExecutionResult with output and trajectory
#         """
#         # Set up working directory
#         if working_dir:
#             work_path = Path(working_dir)
#             ensure_directory(work_path)
#         else:
#             work_path = create_working_directory()

#         # Copy input file if provided
#         if input_file and Path(input_file).exists():
#             copy_file(input_file, work_path)

#         # Copy skills to .claude/skills
#         if skills_dir and Path(skills_dir).exists():
#             claude_skills = work_path / ".claude" / "skills"
#             copy_directory(skills_dir, claude_skills, ignore_patterns=["__pycache__", "*.pyc"])

#         # Prepare environment
#         env = self._prepare_env(work_path)

#         # Build and run command
#         command = self._build_command(prompt)

#         try:
#             result = subprocess.run(
#                 command,
#                 cwd=str(work_path),
#                 env=env,
#                 stdin=subprocess.DEVNULL,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.STDOUT,
#                 text=True,
#                 timeout=self.timeout,
#             )

#             output = self._parse_output(result.stdout or "")
#             trajectory = self._parse_trajectory(env.get("CLAUDE_CONFIG_DIR"))

#             # Calculate duration from trajectory if available
#             duration_ms = 0
#             if trajectory and "events" in trajectory:
#                 for event in trajectory["events"]:
#                     if event.get("type") == "result":
#                         duration_ms = event.get("duration_ms", 0)
#                         break

#             return ExecutionResult(
#                 success=result.returncode == 0,
#                 output=output,
#                 exit_code=result.returncode,
#                 trajectory_data=trajectory,
#                 working_dir=str(work_path),
#                 duration_ms=duration_ms,
#             )

#         except subprocess.TimeoutExpired:
#             return ExecutionResult(
#                 success=False,
#                 output="",
#                 exit_code=-1,
#                 trajectory_data=self._parse_trajectory(env.get("CLAUDE_CONFIG_DIR")),
#                 working_dir=str(work_path),
#                 error=f"Execution timed out after {self.timeout} seconds",
#             )

#         except FileNotFoundError:
#             return ExecutionResult(
#                 success=False,
#                 output="",
#                 exit_code=-1,
#                 working_dir=str(work_path),
#                 error="Claude CLI not found. Ensure ~/.local/bin/claude exists.",
#             )

#         except Exception as e:
#             return ExecutionResult(
#                 success=False,
#                 output="",
#                 exit_code=-1,
#                 working_dir=str(work_path),
#                 error=str(e),
#             )

#     def generate_file_summary(
#         self,
#         file_path: str,
#         max_chars: int = 50,
#         max_read_lines: int = 100,
#         working_dir: str | None = None,
#     ) -> str:
#         """
#         Generate a short summary for a single file.

#         Uses Claude Code CLI to read the file and generate a concise summary.

#         Args:
#             file_path: Path to the file
#             max_chars: Maximum characters for the summary
#             max_read_lines: Maximum lines to read from large files
#             working_dir: Working directory (uses file's directory if None)

#         Returns:
#             File summary string
#         """
#         from .prompt_builder import PromptBuilder

#         # Build summary prompt
#         builder = PromptBuilder()
#         prompt = builder.build_summary_prompt(
#             file_path=file_path,
#             max_chars=max_chars,
#             max_read_lines=max_read_lines,
#         )

#         # Determine working directory
#         if working_dir:
#             work_path = Path(working_dir)
#             ensure_directory(work_path)
#         else:
#             work_path = Path(file_path).parent
#             if not work_path.exists():
#                 work_path = create_working_directory()

#         # Prepare environment (no skills needed for summary)
#         env = self._prepare_env(work_path)

#         # Build command with reduced turns and timeout for summary
#         command = self._build_summary_command(prompt)

#         try:
#             result = subprocess.run(
#                 command,
#                 cwd=str(work_path),
#                 env=env,
#                 stdin=subprocess.DEVNULL,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.STDOUT,
#                 text=True,
#                 timeout=60,  # Short timeout for summary
#             )

#             # Parse the output to extract the summary
#             summary = self._extract_summary_from_output(result.stdout or "")
#             return summary

#         except subprocess.TimeoutExpired:
#             return f"[摘要生成超时，请直接读取文件 {file_path}]"
#         except Exception as e:
#             return f"[摘要生成失败: {str(e)[:50]}，请直接读取文件 {file_path}]"

#     def _build_summary_command(self, instruction: str) -> list[str]:
#         """Build the claude CLI command for summary generation"""
#         claude_path = self._find_claude_path()

#         cmd = [
#             claude_path,
#             "--verbose",
#             "--output-format=stream-json",
#             "--permission-mode=bypassPermissions",
#             "--max-turns", "3",  # Only need a few turns for summary
#             "--print", "--", instruction,
#         ]
#         return cmd

#     def _extract_summary_from_output(self, raw_output: str) -> str:
#         """Extract the summary text from Claude's output"""
#         # Try to find the last assistant message content
#         last_content = ""

#         for line in raw_output.splitlines():
#             stripped = line.strip()
#             if not stripped:
#                 continue

#             try:
#                 obj = json.loads(stripped)
#                 # Look for assistant messages or content
#                 if obj.get("type") == "assistant" and "message" in obj:
#                     message = obj["message"]
#                     if "content" in message:
#                         for block in message["content"]:
#                             if block.get("type") == "text":
#                                 last_content = block.get("text", "")
#                 elif "content" in obj:
#                     last_content = obj["content"]
#             except json.JSONDecodeError:
#                 # Plain text output
#                 if stripped and not stripped.startswith("{"):
#                     last_content = stripped

#         # Clean up the summary
#         summary = last_content.strip()

#         # Remove common prefixes that Claude might add
#         prefixes_to_remove = [
#             "摘要：", "摘要:", "Summary:", "summary:",
#             "文件摘要：", "文件摘要:",
#         ]
#         for prefix in prefixes_to_remove:
#             if summary.startswith(prefix):
#                 summary = summary[len(prefix):].strip()

#         # If empty, return a default message
#         if not summary:
#             return "[无法生成摘要，请直接读取文件]"

#         return summary

#     def execute_with_config(self, config: ExecutionConfig) -> ExecutionResult:
#         """
#         Execute with an ExecutionConfig object.

#         Args:
#             config: Execution configuration

#         Returns:
#             ExecutionResult
#         """
#         from .prompt_builder import PromptBuilder

#         # Build prompt
#         builder = PromptBuilder(config.prompt_config)
#         prompt = builder.build_exploration_prompt(
#             file_path=config.input_file,
#             skills_hint=config.skills_hint,
#             domain=config.domain,
#             output_dir=config.output_dir,
#         )

#         return self.execute(
#             prompt=prompt,
#             working_dir=config.working_dir,
#             skills_dir=config.skills_dir,
#             input_file=config.input_file,
#         )

#     def execute_batch(
#         self,
#         configs: list[ExecutionConfig],
#     ) -> list[ExecutionResult]:
#         """
#         Execute multiple tasks sequentially.

#         Args:
#             configs: List of execution configurations

#         Returns:
#             List of execution results
#         """
#         results = []
#         for config in configs:
#             result = self.execute_with_config(config)
#             results.append(result)
#         return results

#     def _prepare_env(self, working_dir: Path) -> dict[str, str]:
#         """Prepare environment variables for execution"""
#         env = os.environ.copy()

#         # Ensure claude is in PATH
#         env["PATH"] = f"<CLAUDE_CLI_DIR>:{env.get('PATH', '')}"

#         # API configuration
#         if self.api_key:
#             env["ANTHROPIC_API_KEY"] = self.api_key
#             env["ANTHROPIC_AUTH_TOKEN"] = self.api_key

#         if self.base_url:
#             env["ANTHROPIC_BASE_URL"] = self.base_url

#         if self.model:
#             env["ANTHROPIC_MODEL"] = self.model

#         # Model aliases when using custom base URL
#         if env.get("ANTHROPIC_BASE_URL") and env.get("ANTHROPIC_MODEL"):
#             env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = env["ANTHROPIC_MODEL"]
#             env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = env["ANTHROPIC_MODEL"]
#             env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = env["ANTHROPIC_MODEL"]
#             env["CLAUDE_CODE_SUBAGENT_MODEL"] = env["ANTHROPIC_MODEL"]

#         # Runtime settings
#         env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
#         env["IS_SANDBOX"] = "1"
#         env["FORCE_AUTO_BACKGROUND_TASKS"] = "1"
#         env["ENABLE_BACKGROUND_TASKS"] = "1"

#         # Isolated config directory
#         config_dir = working_dir / ".claude_runtime"
#         env["CLAUDE_CONFIG_DIR"] = str(config_dir)

#         # Pre-create required directories
#         for subdir in ["debug", "projects", "shell-snapshots", "statsig", "todos", "skills"]:
#             (config_dir / subdir).mkdir(parents=True, exist_ok=True)

#         return env
    
#     def _find_claude_path(self) -> str:
#         """Always use the user-local Claude CLI executable"""
#         claude_path = Path.home() / ".local" / "bin" / "claude"

#         if claude_path.exists() and os.access(claude_path, os.X_OK):
#             return str(claude_path)

#         raise FileNotFoundError(
#             f"Claude CLI not found or not executable: {claude_path}. "
#             "Please install Claude to ~/.local/bin/claude and ensure it has execute permission."
#         )


#     # def _find_claude_path(self) -> str:
#     #     """Find the claude CLI executable path"""
#     #     import shutil

#     #     # Try to find claude in PATH first
#     #     claude_in_path = shutil.which("claude")
#     #     if claude_in_path:
#     #         return claude_in_path

#     #     # Check common locations
#     #     possible_paths = [
#     #         Path.home() / ".local" / "bin" / "claude",
#     #         Path("<CLAUDE_CLI_DIR>/claude"),
#     #         Path("/usr/local/bin/claude"),
#     #     ]

#     #     for path in possible_paths:
#     #         if path.exists():
#     #             return str(path)

#     #     # Default fallback
#     #     return str(Path.home() / ".local" / "bin" / "claude")

#     def _build_command(self, instruction: str) -> list[str]:
#         """Build the claude CLI command"""
#         claude_path = self._find_claude_path()

#         cmd = [
#             claude_path,
#             "--verbose",
#             "--output-format=stream-json",
#             "--permission-mode=bypassPermissions",
#         ]

#         if self.max_turns:
#             cmd.extend(["--max-turns", str(self.max_turns)])

#         cmd.extend(["--print", "--", instruction])
#         return cmd

#     def _parse_output(self, raw_output: str) -> str:
#         """Parse the raw output from Claude"""
#         parsed_lines = []

#         for line in raw_output.splitlines():
#             stripped = line.strip()
#             if not stripped:
#                 continue

#             try:
#                 obj = json.loads(stripped)
#                 # Extract content from JSON output
#                 if "content" in obj:
#                     parsed_lines.append(obj["content"])
#                 else:
#                     parsed_lines.append(json.dumps(obj, ensure_ascii=False))
#             except json.JSONDecodeError:
#                 parsed_lines.append(stripped)

#         return "\n".join(parsed_lines).strip() or "[No output captured]"

#     def _parse_trajectory(self, config_dir: str | None) -> dict[str, Any] | None:
#         """Parse trajectory from Claude's session files"""
#         if not config_dir:
#             return None

#         projects_dir = Path(config_dir) / "projects"
#         if not projects_dir.exists():
#             return None

#         # Find most recent project directory
#         project_dirs = sorted(
#             [d for d in projects_dir.iterdir() if d.is_dir()],
#             key=lambda x: x.stat().st_mtime,
#             reverse=True,
#         )

#         if not project_dirs:
#             return None

#         latest_dir = project_dirs[0]
#         jsonl_files = list(latest_dir.glob("*.jsonl"))

#         if not jsonl_files:
#             return None

#         # Parse all events from JSONL files
#         events = []
#         for jsonl_file in jsonl_files:
#             try:
#                 with open(jsonl_file, "r", encoding="utf-8") as f:
#                     for line in f:
#                         line = line.strip()
#                         if not line:
#                             continue
#                         try:
#                             events.append(json.loads(line))
#                         except json.JSONDecodeError:
#                             pass
#             except Exception:
#                 pass

#         if not events:
#             return None

#         return {
#             "session_id": latest_dir.name,
#             "events_count": len(events),
#             "events": events,
#         }



import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import ExecutionConfig, PipelineConfig
from ..utils.file_utils import (
    copy_directory,
    copy_file,
    create_working_directory,
    ensure_directory,
)


@dataclass
class ExecutionResult:
    """Result of Claude Code execution"""
    success: bool
    output: str
    exit_code: int
    trajectory_data: dict[str, Any] | None = None
    working_dir: str | None = None
    duration_ms: int = 0
    error: str | None = None

    @property
    def has_trajectory(self) -> bool:
        return self.trajectory_data is not None and len(self.trajectory_data.get("events", [])) > 0


class ClaudeExecutor:
    """Executor for running Claude Code tasks"""

    def __init__(
        self,
        model: str = "claude-opus-4-5-20251124",
        max_turns: int = 20,
        timeout: int = 300,
        api_key: str | None = None,
        base_url: str | None = None,
        conda_env: str | None = None,
    ):
        """
        Initialize Claude executor.

        Args:
            model: Model name to use
            max_turns: Maximum conversation turns
            timeout: Execution timeout in seconds
            api_key: API key (defaults to env var)
            base_url: API base URL (defaults to env var)
            conda_env: Conda environment name (defaults to 'base')
        """
        self.model = model
        self.max_turns = max_turns
        self.timeout = timeout
        self.api_key = (
            api_key
            or os.environ.get("ANTHROPIC_AUTH_TOKEN")
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL")
        self.conda_env = conda_env or "base"

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "ClaudeExecutor":
        """Create executor from pipeline config"""
        return cls(
            model=config.model,
            max_turns=config.max_turns,
            timeout=config.timeout,
            conda_env=config.conda_env,
        )

    def execute(
        self,
        prompt: str,
        working_dir: str | None = None,
        skills_dir: str | None = None,
        input_file: str | None = None,
    ) -> ExecutionResult:
        """
        Execute a Claude Code task.

        Args:
            prompt: The prompt/instruction for Claude
            working_dir: Working directory (uses input file dir if None)
            skills_dir: Skills directory to copy
            input_file: Input file path

        Returns:
            ExecutionResult with output and trajectory
        """
        resolved_input_file: Path | None = None
        if input_file and Path(input_file).exists():
            resolved_input_file = Path(input_file).resolve()

        # Set up working directory
        if working_dir:
            work_path = Path(working_dir).resolve()
            ensure_directory(work_path)
        elif resolved_input_file:
            # Prefer the real input file directory so Claude can access the file naturally
            work_path = resolved_input_file.parent
        else:
            work_path = create_working_directory()

        # Copy input file only if we're not already working in its parent directory
        if resolved_input_file and work_path.resolve() != resolved_input_file.parent:
            copy_file(str(resolved_input_file), work_path)

        # Copy skills to .claude/skills
        if skills_dir and Path(skills_dir).exists():
            claude_skills = work_path / ".claude" / "skills"
            copy_directory(
                skills_dir,
                claude_skills,
                ignore_patterns=["__pycache__", "*.pyc"],
            )

        # Prepare environment
        env = self._prepare_env(work_path)

        # Build and run command
        # For long prompts, use a temporary file to avoid "Argument list too long" error
        # Linux ARG_MAX is typically 128KB-2MB, use 50KB as safe threshold
        prompt_bytes = len(prompt.encode('utf-8'))
        use_prompt_file = prompt_bytes > 50000
        prompt_file_path = None

        try:
            if use_prompt_file:
                # Write prompt to temporary file
                import tempfile
                prompt_file = tempfile.NamedTemporaryFile(
                    mode='w',
                    suffix='.txt',
                    dir=str(work_path),
                    delete=False,
                    encoding='utf-8',
                )
                prompt_file.write(prompt)
                prompt_file.close()
                prompt_file_path = prompt_file.name

                # Use shell command to read file and pass to claude
                # Format: cat prompt.txt | claude --print -p -
                # Or use bash -c with heredoc
                claude_path = self._find_claude_path()
                shell_cmd = f'cat "{prompt_file_path}" | {claude_path} --verbose --output-format=stream-json --permission-mode=bypassPermissions'
                if self.model:
                    shell_cmd += f' --model {self.model}'
                if self.max_turns:
                    shell_cmd += f' --max-turns {self.max_turns}'
                shell_cmd += ' --print -p -'

                if self.conda_env and self.conda_env != "base":
                    command = ["conda", "run", "-n", self.conda_env, "--no-capture-output", "bash", "-c", shell_cmd]
                else:
                    command = ["bash", "-c", shell_cmd]
            else:
                command = self._build_command(prompt)

            result = subprocess.run(
                command,
                cwd=str(work_path),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self.timeout,
            )

            # Clean up prompt file
            if prompt_file_path and Path(prompt_file_path).exists():
                Path(prompt_file_path).unlink()

            output = self._parse_output(result.stdout or "")
            trajectory = self._parse_trajectory(env.get("CLAUDE_CONFIG_DIR"))

            # Calculate duration from trajectory if available
            duration_ms = 0
            if trajectory and "events" in trajectory:
                for event in trajectory["events"]:
                    if event.get("type") == "result":
                        duration_ms = event.get("duration_ms", 0)
                        break

            # Capture error message when execution fails
            error_msg = None
            if result.returncode != 0:
                raw_output = (result.stdout or "").strip()
                error_msg = raw_output[:1000] if raw_output else f"Exit code: {result.returncode}"

            return ExecutionResult(
                success=result.returncode == 0,
                output=output,
                exit_code=result.returncode,
                trajectory_data=trajectory,
                working_dir=str(work_path),
                duration_ms=duration_ms,
                error=error_msg,
            )

        except subprocess.TimeoutExpired:
            # Clean up prompt file on timeout
            if prompt_file_path and Path(prompt_file_path).exists():
                Path(prompt_file_path).unlink()
            return ExecutionResult(
                success=False,
                output="",
                exit_code=-1,
                trajectory_data=self._parse_trajectory(env.get("CLAUDE_CONFIG_DIR")),
                working_dir=str(work_path),
                error=f"Execution timed out after {self.timeout} seconds",
            )

        except FileNotFoundError as e:
            # Clean up prompt file on error
            if prompt_file_path and Path(prompt_file_path).exists():
                Path(prompt_file_path).unlink()
            return ExecutionResult(
                success=False,
                output="",
                exit_code=-1,
                working_dir=str(work_path),
                error=str(e),
            )

        except Exception as e:
            # Clean up prompt file on error
            if prompt_file_path and Path(prompt_file_path).exists():
                Path(prompt_file_path).unlink()
            return ExecutionResult(
                success=False,
                output="",
                exit_code=-1,
                working_dir=str(work_path),
                error=str(e),
            )

    def generate_file_summary(
        self,
        file_path: str,
        max_chars: int = 50,
        max_read_lines: int = 100,
        working_dir: str | None = None,
        timeout: int = 180,
        max_retries: int = 3,
    ) -> str:
        """
        Generate a short summary for a single file.

        Uses Claude Code CLI to read the file and generate a concise summary.

        Args:
            file_path: Path to the file
            max_chars: Maximum characters for the summary
            max_read_lines: Maximum lines to read from large files
            working_dir: Working directory (uses file's directory if None)
            timeout: Timeout in seconds for each attempt (default: 180)
            max_retries: Maximum number of retry attempts (default: 3)

        Returns:
            File summary string
        """
        import time
        from .prompt_builder import PromptBuilder

        resolved_file_path = str(Path(file_path).resolve())

        # Build summary prompt
        builder = PromptBuilder()
        prompt = builder.build_summary_prompt(
            file_path=resolved_file_path,
            max_chars=max_chars,
            max_read_lines=max_read_lines,
        )

        # Determine working directory
        if working_dir:
            work_path = Path(working_dir).resolve()
            ensure_directory(work_path)
        else:
            work_path = Path(resolved_file_path).parent
            if not work_path.exists():
                work_path = create_working_directory()

        # Prepare environment (no skills needed for summary)
        env = self._prepare_env(work_path)

        # Build command with reduced turns and timeout for summary
        command = self._build_summary_command(prompt)

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                result = subprocess.run(
                    command,
                    cwd=str(work_path),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout,
                )

                summary = self._extract_summary_from_output(result.stdout or "")
                if summary:
                    return summary
                # Empty summary, retry
                last_error = "empty_summary"

            except subprocess.TimeoutExpired:
                last_error = "timeout"
                if attempt < max_retries:
                    print(f"[Executor] Timeout on attempt {attempt}/{max_retries}, retrying...")
                    time.sleep(2)  # Brief pause before retry
                continue

            except Exception as e:
                last_error = str(e)[:50]
                if attempt < max_retries:
                    print(f"[Executor] Error on attempt {attempt}/{max_retries}: {last_error}, retrying...")
                    time.sleep(2)
                continue

        # All retries exhausted
        if last_error == "timeout":
            return f"[摘要生成超时(重试{max_retries}次)，请直接读取文件 {resolved_file_path}]"
        else:
            return f"[摘要生成失败: {last_error}，请直接读取文件 {resolved_file_path}]"

    def _build_summary_command(self, instruction: str) -> list[str]:
        """Build the claude CLI command for summary generation, wrapped in conda run if needed"""
        claude_path = self._find_claude_path()

        claude_args = [
            claude_path,
            "--verbose",
            "--output-format=stream-json",
            "--permission-mode=bypassPermissions",
        ]

        if self.model:
            claude_args.extend(["--model", self.model])

        claude_args.extend([
            "--max-turns",
            "3",
            "--print",
            "--",
            instruction,
        ])

        # Wrap with conda run for non-base environments
        if self.conda_env and self.conda_env != "base":
            cmd = [
                "conda", "run", "-n", self.conda_env, "--no-capture-output",
            ] + claude_args
        else:
            cmd = claude_args

        return cmd

    def _extract_summary_from_output(self, raw_output: str) -> str:
        """Extract the summary text from Claude's output"""
        last_content = ""

        for line in raw_output.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            try:
                obj = json.loads(stripped)
                # Only process dict objects
                if not isinstance(obj, dict):
                    continue
                if obj.get("type") == "assistant" and "message" in obj:
                    message = obj["message"]
                    if isinstance(message, dict) and "content" in message:
                        for block in message["content"]:
                            if isinstance(block, dict) and block.get("type") == "text":
                                last_content = block.get("text", "")
                elif "content" in obj:
                    last_content = str(obj["content"])
            except json.JSONDecodeError:
                if stripped and not stripped.startswith("{"):
                    last_content = stripped

        summary = last_content.strip()

        prefixes_to_remove = [
            "摘要：", "摘要:", "Summary:", "summary:",
            "文件摘要：", "文件摘要:",
        ]
        for prefix in prefixes_to_remove:
            if summary.startswith(prefix):
                summary = summary[len(prefix):].strip()

        if not summary:
            return "[无法生成摘要，请直接读取文件]"

        return summary

    def execute_with_config(self, config: ExecutionConfig) -> ExecutionResult:
        """
        Execute with an ExecutionConfig object.

        Args:
            config: Execution configuration

        Returns:
            ExecutionResult
        """
        from .prompt_builder import PromptBuilder

        resolved_input_file = (
            str(Path(config.input_file).resolve())
            if config.input_file
            else None
        )

        builder = PromptBuilder(config.prompt_config)
        prompt = builder.build_exploration_prompt(
            file_path=resolved_input_file,
            skills_hint=config.skills_hint,
            domain=config.domain,
            output_dir=config.output_dir,
        )

        return self.execute(
            prompt=prompt,
            working_dir=config.working_dir,
            skills_dir=config.skills_dir,
            input_file=resolved_input_file,
        )

    def execute_batch(
        self,
        configs: list[ExecutionConfig],
    ) -> list[ExecutionResult]:
        """
        Execute multiple tasks sequentially.

        Args:
            configs: List of execution configurations

        Returns:
            List of execution results
        """
        results = []
        for config in configs:
            result = self.execute_with_config(config)
            results.append(result)
        return results

    # def _prepare_env(self, working_dir: Path) -> dict[str, str]:
    #     """Prepare environment variables for execution"""
    #     env = os.environ.copy()

    #     # Prefer the actual Claude CLI location in this environment
    #     cli_bin_dir = "<CLAUDE_CLI_DIR>"
    #     env["PATH"] = f"{cli_bin_dir}:{env.get('PATH', '')}"

    #     # API configuration
    #     if self.api_key:
    #         env["ANTHROPIC_API_KEY"] = self.api_key
    #         env["ANTHROPIC_AUTH_TOKEN"] = self.api_key

    #     if self.base_url:
    #         env["ANTHROPIC_BASE_URL"] = self.base_url

    #     if self.model:
    #         env["ANTHROPIC_MODEL"] = self.model

    #     # Model aliases when using custom base URL
    #     if env.get("ANTHROPIC_BASE_URL") and env.get("ANTHROPIC_MODEL"):
    #         env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = env["ANTHROPIC_MODEL"]
    #         env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = env["ANTHROPIC_MODEL"]
    #         env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = env["ANTHROPIC_MODEL"]
    #         env["CLAUDE_CODE_SUBAGENT_MODEL"] = env["ANTHROPIC_MODEL"]

    #     # Runtime settings
    #     env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    #     env["IS_SANDBOX"] = "1"
    #     env["FORCE_AUTO_BACKGROUND_TASKS"] = "1"
    #     env["ENABLE_BACKGROUND_TASKS"] = "1"

    #     # Isolated config directory
    #     config_dir = working_dir / ".claude_runtime"
    #     env["CLAUDE_CONFIG_DIR"] = str(config_dir)

    #     # Pre-create required directories
    #     for subdir in ["debug", "projects", "shell-snapshots", "statsig", "todos", "skills"]:
    #         (config_dir / subdir).mkdir(parents=True, exist_ok=True)

    #     return env
    def _prepare_env(self, working_dir: Path) -> dict[str, str]:
        """Prepare environment variables for execution"""
        env = os.environ.copy()

        # Determine conda environment paths
        conda_env = self.conda_env or "base"
        if conda_env == "base":
            conda_prefix = "/opt/conda"
            conda_bin_dir = "/opt/conda/bin"
        else:
            conda_prefix = f"/opt/conda/envs/{conda_env}"
            conda_bin_dir = f"/opt/conda/envs/{conda_env}/bin"

        env["PATH"] = f"{conda_bin_dir}:{env.get('PATH', '')}"

        # Hint child processes to use specified conda environment
        env["PYTHONEXECUTABLE"] = f"{conda_bin_dir}/python"
        env["CONDA_PREFIX"] = conda_prefix
        env["CONDA_DEFAULT_ENV"] = conda_env

        # Log conda environment for verification
        print(f"[Executor] Conda env: {conda_env}, prefix: {conda_prefix}")

        # API configuration
        if self.api_key:
            env["ANTHROPIC_API_KEY"] = self.api_key
            env["ANTHROPIC_AUTH_TOKEN"] = self.api_key

        if self.base_url:
            env["ANTHROPIC_BASE_URL"] = self.base_url

        if self.model:
            env["ANTHROPIC_MODEL"] = self.model

        # Model aliases when using custom base URL
        if env.get("ANTHROPIC_BASE_URL") and env.get("ANTHROPIC_MODEL"):
            env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = env["ANTHROPIC_MODEL"]
            env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = env["ANTHROPIC_MODEL"]
            env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = env["ANTHROPIC_MODEL"]
            env["CLAUDE_CODE_SUBAGENT_MODEL"] = env["ANTHROPIC_MODEL"]

        # Runtime settings
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        env["IS_SANDBOX"] = "1"
        env["FORCE_AUTO_BACKGROUND_TASKS"] = "1"
        env["ENABLE_BACKGROUND_TASKS"] = "1"

        # Isolated config directory
        config_dir = working_dir / ".claude_runtime"
        env["CLAUDE_CONFIG_DIR"] = str(config_dir)

        # Pre-create required directories
        for subdir in ["debug", "projects", "shell-snapshots", "statsig", "todos", "skills"]:
            (config_dir / subdir).mkdir(parents=True, exist_ok=True)

        return env


    def _find_claude_path(self) -> str:
        """Find the actual Claude CLI executable path for this environment"""
        possible_paths = [
            Path.home() / ".local" / "bin" / "claude",
            Path("/usr/local/bin/claude"),
            Path("/usr/bin/claude"),
        ]

        for path in possible_paths:
            if path.exists() and os.access(path, os.X_OK):
                return str(path)

        raise FileNotFoundError(
            "Claude CLI not found. Checked: "
            + ", ".join(str(p) for p in possible_paths)
        )

    def _build_command(self, instruction: str) -> list[str]:
        """Build the claude CLI command, wrapped in conda run if needed.

        Args:
            instruction: The prompt/instruction

        Returns:
            Command list for subprocess
        """
        claude_path = self._find_claude_path()

        claude_args = [
            claude_path,
            "--verbose",
            "--output-format=stream-json",
            "--permission-mode=bypassPermissions",
        ]

        if self.model:
            claude_args.extend(["--model", self.model])

        if self.max_turns:
            claude_args.extend(["--max-turns", str(self.max_turns)])

        claude_args.extend(["--print", "--", instruction])

        # Wrap with conda run for non-base environments
        if self.conda_env and self.conda_env != "base":
            cmd = [
                "conda", "run", "-n", self.conda_env, "--no-capture-output",
            ] + claude_args
        else:
            cmd = claude_args

        return cmd

    def _parse_output(self, raw_output: str) -> str:
        """Parse the raw output from Claude"""
        parsed_lines = []

        for line in raw_output.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            try:
                obj = json.loads(stripped)
                # Only check for "content" key if obj is a dict
                if isinstance(obj, dict) and "content" in obj:
                    parsed_lines.append(str(obj["content"]))
                elif isinstance(obj, dict):
                    parsed_lines.append(json.dumps(obj, ensure_ascii=False))
                else:
                    # obj is not a dict (e.g., int, list, str)
                    parsed_lines.append(str(obj))
            except json.JSONDecodeError:
                parsed_lines.append(stripped)

        return "\n".join(parsed_lines).strip() or "[No output captured]"

    def _parse_trajectory(self, config_dir: str | None) -> dict[str, Any] | None:
        """Parse trajectory from Claude's session files"""
        if not config_dir:
            return None

        projects_dir = Path(config_dir) / "projects"
        if not projects_dir.exists():
            return None

        project_dirs = sorted(
            [d for d in projects_dir.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        if not project_dirs:
            return None

        latest_dir = project_dirs[0]
        jsonl_files = list(latest_dir.glob("*.jsonl"))

        if not jsonl_files:
            return None

        events = []
        for jsonl_file in jsonl_files:
            try:
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            except Exception:
                pass

        if not events:
            return None

        return {
            "session_id": latest_dir.name,
            "events_count": len(events),
            "events": events,
        }
