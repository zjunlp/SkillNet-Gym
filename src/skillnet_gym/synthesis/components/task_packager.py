"""Task packager for generating complete Harbor task directories"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import HarborTask, ProcessedTrajectory, Trajectory
from ..utils.file_utils import (
    copy_directory,
    copy_file,
    ensure_directory,
    write_file_content,
)
from ..utils.dependency_extractor import DependencyExtractor, ExtractedDependencies
from ..utils.path_normalizer import PathNormalizer
from .test_generator import TestGenerator


# task.toml template
TASK_TOML_TEMPLATE = '''version = "1.0"

[metadata]
author_name = "{author_name}"
difficulty = "{difficulty}"
category = "{category}"
tags = {tags}

[verifier]
timeout_sec = {verifier_timeout}

[agent]
timeout_sec = {agent_timeout}

[environment]
build_timeout_sec = {build_timeout}
cpus = {cpus}
memory_mb = {memory_mb}
'''

# Dockerfile template - enhanced with apt dependencies
DOCKERFILE_TEMPLATE = '''FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y python3 python3-pip {apt_packages}

WORKDIR /root

# Python dependencies
{pip_install_commands}

# Input files
{copy_input_files}

# Copy skills to agent-specific locations
# Claude Code
COPY skills /root/.claude/skills
# Codex
COPY skills /root/.codex/skills
# OpenCode - singular "skill"
COPY skills /root/.opencode/skill
# Goose
COPY skills /root/.goose/skills
# Factory
COPY skills /root/.factory/skills
# Portable agents format (Goose, Amp)
COPY skills /root/.agents/skills
# Gemini
COPY skills /root/.gemini/skills
'''

# solve.sh template
SOLVE_SH_TEMPLATE = '''#!/bin/bash
set -e

# Auto-generated solution based on execution trajectory
# This script recreates the outputs produced during synthesis

{script_content}
'''


class TaskPackager:
    """Generates complete Harbor task packages"""

    def __init__(
        self,
        author_name: str = "Auto-Synthesis Pipeline",
        default_difficulty: str = "medium",
        default_category: str = "general",
    ):
        """
        Initialize the packager.

        Args:
            author_name: Default author name for tasks
            default_difficulty: Default difficulty level
            default_category: Default category
        """
        self.author_name = author_name
        self.default_difficulty = default_difficulty
        self.default_category = default_category
        self.test_generator = TestGenerator()
        self.dependency_extractor = DependencyExtractor()

    def package(
        self,
        task_id: str | None,
        trajectory: Trajectory,
        instruction: str,
        tests: str | None = None,
        input_files: list[str] | None = None,
        used_skills: list[str] | None = None,
        pip_packages: list[str] | None = None,
        skills_dir: str | None = None,
        output_dir: str = "./workspaces",
        metadata: dict[str, Any] | None = None,
    ) -> HarborTask:
        """
        Generate a complete Harbor task package.

        Args:
            task_id: Task identifier (generated if None)
            trajectory: Execution trajectory
            instruction: Task instruction content
            tests: Test file content (generated if None)
            input_files: Input file paths to include
            used_skills: Skills used in the task
            pip_packages: Python packages to install
            skills_dir: Path to skills directory to copy
            output_dir: Base output directory
            metadata: Additional metadata

        Returns:
            HarborTask object with task information
        """
        # Generate task ID if not provided
        if not task_id:
            task_id = self._generate_task_id()

        # Set up task directory structure
        task_dir = Path(output_dir) / task_id
        env_dir = task_dir / "environment"
        tests_dir = task_dir / "tests"
        solution_dir = task_dir / "solution"

        ensure_directory(task_dir)
        ensure_directory(env_dir)
        ensure_directory(tests_dir)
        ensure_directory(solution_dir)

        # Prepare metadata
        meta = metadata or {}
        difficulty = meta.get("difficulty", self.default_difficulty)
        category = meta.get("category", self.default_category)
        tags = meta.get("tags", self._infer_tags(trajectory, used_skills))

        # Generate task.toml
        task_toml = self._generate_task_toml(
            difficulty=difficulty,
            category=category,
            tags=tags,
        )
        write_file_content(task_dir / "task.toml", task_toml)

        # Initialize path normalizer for consistent container paths
        path_normalizer = PathNormalizer(
            input_files=input_files or [],
            container_workdir="/root",
        )

        # Normalize and write instruction.md
        normalized_instruction = path_normalizer.normalize_instruction(instruction)
        write_file_content(task_dir / "instruction.md", normalized_instruction)

        # Extract dependencies from all sources
        extracted_deps = self._extract_all_dependencies(
            skills_dir=skills_dir,
            trajectory=trajectory,
            pip_packages=pip_packages,
        )

        # Generate and write Dockerfile with extracted dependencies
        dockerfile = self._generate_dockerfile(
            input_files=input_files or [],
            extracted_deps=extracted_deps,
        )
        write_file_content(env_dir / "Dockerfile", dockerfile)

        # Write extracted dependencies to JSON for debugging
        deps_json = json.dumps(extracted_deps.to_dict(), indent=2, ensure_ascii=False)
        write_file_content(env_dir / "extracted_dependencies.json", deps_json)

        # Copy input files to environment
        actual_input_files = []
        if input_files:
            for input_file in input_files:
                src = Path(input_file)
                if src.exists():
                    dst = env_dir / src.name
                    copy_file(src, dst)
                    actual_input_files.append(str(dst))

        # Copy skills to environment
        if skills_dir:
            skills_path = Path(skills_dir)
            if skills_path.exists():
                env_skills = env_dir / "skills"
                copy_directory(skills_path, env_skills, ignore_patterns=["__pycache__", "*.pyc"])
            elif used_skills:
                # Create skills directory structure for specific skills
                self._copy_specific_skills(skills_path.parent, used_skills, env_dir / "skills")

        # Generate and write tests (with path normalization)
        if tests is None:
            tests = self.test_generator.generate(trajectory)
        normalized_tests = path_normalizer.normalize_tests(tests)
        write_file_content(tests_dir / "test_outputs.py", normalized_tests)

        # Write test.sh
        test_sh = self.test_generator.generate_test_sh()
        write_file_content(tests_dir / "test.sh", test_sh)

        # Generate and write solve.sh (oracle solution, with path normalization)
        solve_sh = self._generate_solve_sh(trajectory)
        normalized_solve_sh = path_normalizer.normalize_solve_sh(solve_sh)
        write_file_content(solution_dir / "solve.sh", normalized_solve_sh)

        # Write trajectory.json (保存完整探索轨迹)
        trajectory_data = self._serialize_trajectory(trajectory)
        write_file_content(task_dir / "trajectory.json", json.dumps(trajectory_data, indent=2, ensure_ascii=False))

        return HarborTask(
            task_id=task_id,
            task_path=str(task_dir),
            instruction=normalized_instruction,
            input_files=actual_input_files,
            output_files=trajectory.output_files,
            used_skills=used_skills or [],
            metadata={
                "difficulty": difficulty,
                "category": category,
                "tags": tags,
            },
        )

    def package_from_processed(
        self,
        processed: ProcessedTrajectory,
        instruction: str,
        skills_dir: str | None = None,
        output_dir: str = "./workspaces",
        task_id: str | None = None,
    ) -> HarborTask:
        """
        Package a task from a ProcessedTrajectory.

        Args:
            processed: Processed trajectory
            instruction: Task instruction content
            skills_dir: Path to skills directory
            output_dir: Output directory for tasks
            task_id: Optional task ID

        Returns:
            HarborTask object
        """
        return self.package(
            task_id=task_id,
            trajectory=processed.trajectory,
            instruction=instruction,
            input_files=processed.trajectory.input_files,
            used_skills=processed.used_skills,
            pip_packages=processed.pip_packages,
            skills_dir=skills_dir,
            output_dir=output_dir,
            metadata=processed.metadata,
        )

    def _generate_task_id(self) -> str:
        """Generate a unique task ID"""
        timestamp = datetime.now().strftime("%Y%m%d")
        unique = uuid.uuid4().hex[:8]
        return f"task_{timestamp}_{unique}"

    def _generate_task_toml(
        self,
        difficulty: str = "medium",
        category: str = "general",
        tags: list[str] | None = None,
    ) -> str:
        """Generate task.toml content"""
        tags = tags or []
        tags_str = json.dumps(tags)

        return TASK_TOML_TEMPLATE.format(
            author_name=self.author_name,
            difficulty=difficulty,
            category=category,
            tags=tags_str,
            verifier_timeout=300.0,
            agent_timeout=600.0,
            build_timeout=300.0,
            cpus=1,
            memory_mb=4096,
        )

    def _extract_all_dependencies(
        self,
        skills_dir: str | None,
        trajectory: Trajectory,
        pip_packages: list[str] | None = None,
    ) -> ExtractedDependencies:
        """
        Extract dependencies from all available sources.

        Sources:
        1. Skill directory analysis (requirements.txt, dependencies.json, SKILL.md)
        2. Trajectory analysis (pip/apt/conda commands, import statements)
        3. Explicitly provided pip_packages

        Args:
            skills_dir: Path to skills directory
            trajectory: Execution trajectory
            pip_packages: Explicitly provided pip packages

        Returns:
            ExtractedDependencies instance
        """
        deps = ExtractedDependencies()

        # 1. Extract from skills directory
        if skills_dir:
            skill_deps = self.dependency_extractor.extract_from_skills_dir(skills_dir)
            deps.merge_from(skill_deps)

        # 2. Extract from trajectory
        traj_deps = self.dependency_extractor.extract_from_trajectory(trajectory)
        deps.merge_from(traj_deps)

        # 3. Add explicitly provided packages
        if pip_packages:
            deps.pip_packages.update(pip_packages)
            deps.sources["explicit"] = list(pip_packages)

        return deps

    def _generate_dockerfile(
        self,
        input_files: list[str],
        extracted_deps: ExtractedDependencies,
    ) -> str:
        """
        Generate Dockerfile content with extracted dependencies.

        Args:
            input_files: List of input file paths
            extracted_deps: Extracted dependencies from all sources

        Returns:
            Dockerfile content string
        """
        # Generate apt packages string (for inline in RUN command)
        apt_packages_str = ""
        if extracted_deps.apt_packages:
            apt_packages_str = " ".join(sorted(extracted_deps.apt_packages))

        # Generate pip install commands
        pip_commands = ""
        if extracted_deps.pip_packages:
            packages_str = " ".join(sorted(extracted_deps.pip_packages))
            pip_commands = f"RUN pip3 install --break-system-packages {packages_str}"

        # Generate COPY commands for input files
        copy_commands = []
        for input_file in input_files:
            filename = Path(input_file).name
            copy_commands.append(f"COPY {filename} /root/")

        copy_str = "\n".join(copy_commands) if copy_commands else "# No input files"

        return DOCKERFILE_TEMPLATE.format(
            apt_packages=apt_packages_str,
            pip_install_commands=pip_commands,
            copy_input_files=copy_str,
        )

    def _generate_solve_sh(self, trajectory: Trajectory) -> str:
        """Generate solve.sh based on trajectory"""
        script_lines = []

        # Extract file creation operations
        for step in trajectory.steps:
            if step.action_type != "tool_use":
                continue

            if step.tool_name == "Write" and step.tool_input:
                file_path = step.tool_input.get("file_path", "")
                content = step.tool_input.get("content", "")

                if file_path and content:
                    # Escape content for heredoc
                    escaped_content = content.replace("'", "'\"'\"'")

                    script_lines.append(f"# Create {file_path}")
                    script_lines.append(f"mkdir -p $(dirname {file_path})")
                    script_lines.append(f"cat << 'HEREDOC_EOF' > {file_path}")
                    script_lines.append(content)
                    script_lines.append("HEREDOC_EOF")
                    script_lines.append("")

            elif step.tool_name == "Bash" and step.tool_input:
                command = step.tool_input.get("command", "")
                if command:
                    script_lines.append(f"# Execute command")
                    script_lines.append(command)
                    script_lines.append("")

        if not script_lines:
            script_lines.append("echo 'No automated solution available'")
            script_lines.append("echo 'Please implement the task manually'")

        return SOLVE_SH_TEMPLATE.format(
            script_content="\n".join(script_lines),
        )

    def _infer_tags(
        self,
        trajectory: Trajectory,
        used_skills: list[str] | None,
    ) -> list[str]:
        """Infer appropriate tags from trajectory"""
        tags = set()

        # Add skill-based tags
        if used_skills:
            for skill in used_skills:
                tags.add(skill)

        # Add file type tags
        for file_path in trajectory.output_files:
            ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
            if ext in ("json", "csv", "xlsx"):
                tags.add("data-processing")
            elif ext in ("md", "txt"):
                tags.add("text-processing")
            elif ext in ("pdf",):
                tags.add("document-analysis")
            elif ext in ("py", "js", "ts"):
                tags.add("code-generation")

        # Add tool-based tags
        for step in trajectory.steps:
            if step.tool_name == "Bash":
                tags.add("shell")

        return list(tags)[:5]  # Limit to 5 tags

    def _copy_specific_skills(
        self,
        skills_base: Path,
        skill_names: list[str],
        dest_dir: Path,
    ):
        """Copy specific skills to destination"""
        ensure_directory(dest_dir)

        for skill_name in skill_names:
            skill_src = skills_base / skill_name
            if skill_src.exists():
                skill_dst = dest_dir / skill_name
                copy_directory(skill_src, skill_dst, ignore_patterns=["__pycache__", "*.pyc"])

    def _serialize_trajectory(self, trajectory: Trajectory) -> dict[str, Any]:
        """Serialize Trajectory object to JSON-compatible dict"""
        return {
            "session_id": trajectory.session_id,
            "model": trajectory.model,
            "success": trajectory.success,
            "duration_ms": trajectory.duration_ms,
            "input_files": trajectory.input_files,
            "output_files": trajectory.output_files,
            "num_steps": trajectory.num_steps,
            "steps": [
                {
                    "step_id": step.step_id,
                    "action_type": step.action_type,
                    "tool_name": step.tool_name,
                    "tool_input": step.tool_input,
                    "tool_output": step.tool_output,
                    "timestamp": step.timestamp,
                    "reasoning": step.reasoning,
                }
                for step in trajectory.steps
            ],
            "raw_events": trajectory.raw_events,
        }

    def package_skillsbench_format(
        self,
        task_id: str,
        instruction: str,
        tests_content: str,
        solve_sh_content: str,
        input_files: list[str],
        skills_dir: str | None = None,
        output_dir: str = "./workspaces",
        metadata: dict[str, Any] | None = None,
        pip_packages: list[str] | None = None,
        apt_packages: list[str] | None = None,
        used_skills: list[str] | None = None,
        output_files: list[str] | None = None,
    ) -> HarborTask:
        """
        Package task in skillsbench format.

        Output structure:
        task_id/
        ├── task.toml
        ├── instruction.md
        ├── environment/
        │   ├── Dockerfile
        │   ├── skills/
        │   └── <input-files>
        ├── solution/
        │   └── solve.sh
        └── tests/
            ├── test.sh
            └── test_outputs.py

        Args:
            task_id: Unique task identifier
            instruction: Task instruction content
            tests_content: Pytest test file content (already path-normalized)
            solve_sh_content: solve.sh script content (already path-normalized)
            input_files: List of input file paths to copy
            skills_dir: Path to skills directory to copy
            output_dir: Base output directory
            metadata: Additional task metadata
            pip_packages: Python packages to install
            apt_packages: System packages to install
            used_skills: List of skill names used in the task
            output_files: List of output file paths created by the task

        Returns:
            HarborTask object with task information
        """
        task_path = Path(output_dir) / task_id

        # Create directory structure
        env_dir = task_path / "environment"
        solution_dir = task_path / "solution"
        tests_dir = task_path / "tests"

        ensure_directory(task_path)
        ensure_directory(env_dir)
        ensure_directory(solution_dir)
        ensure_directory(tests_dir)

        # Prepare metadata
        meta = metadata or {}
        difficulty = meta.get("difficulty", self.default_difficulty)
        category = meta.get("category", self.default_category)
        tags = meta.get("tags", [])

        # 1. Write task.toml
        task_toml = self._generate_task_toml(
            difficulty=difficulty,
            category=category,
            tags=tags,
        )
        write_file_content(task_path / "task.toml", task_toml)

        # 2. Write instruction.md
        write_file_content(task_path / "instruction.md", instruction)

        # 3. Setup environment/
        #    - Copy input files
        actual_input_files = []
        for input_file in input_files:
            src = Path(input_file)
            if src.exists():
                dst = env_dir / src.name
                copy_file(src, dst)
                actual_input_files.append(str(dst))

        #    - Copy skills
        if skills_dir:
            skills_path = Path(skills_dir)
            if skills_path.exists():
                env_skills = env_dir / "skills"
                copy_directory(
                    skills_path,
                    env_skills,
                    ignore_patterns=["__pycache__", "*.pyc"],
                )
            else:
                # Create empty skills directory
                ensure_directory(env_dir / "skills")
        else:
            # Create empty skills directory
            ensure_directory(env_dir / "skills")

        #    - Generate Dockerfile
        extracted_deps = ExtractedDependencies()
        if pip_packages:
            extracted_deps.pip_packages.update(pip_packages)
        if apt_packages:
            extracted_deps.apt_packages.update(apt_packages)

        dockerfile = self._generate_dockerfile(
            input_files=input_files,
            extracted_deps=extracted_deps,
        )
        write_file_content(env_dir / "Dockerfile", dockerfile)

        # 4. Write solution/solve.sh
        write_file_content(solution_dir / "solve.sh", solve_sh_content)

        # 5. Write tests/test.sh
        test_sh = self._generate_skillsbench_test_sh()
        write_file_content(tests_dir / "test.sh", test_sh)

        # 6. Write tests/test_outputs.py
        write_file_content(tests_dir / "test_outputs.py", tests_content)

        return HarborTask(
            task_id=task_id,
            task_path=str(task_path),
            instruction=instruction,
            input_files=actual_input_files,
            output_files=output_files or [],
            used_skills=used_skills or [],
            metadata={
                "difficulty": difficulty,
                "category": category,
                "tags": tags,
                "format": "skillsbench",
            },
        )

    def _generate_skillsbench_test_sh(self) -> str:
        """Generate standard test.sh for skillsbench format."""
        return '''#!/bin/bash
pip3 install --break-system-packages pytest==8.4.1 pytest-json-ctrf==0.3.5
mkdir -p /logs/verifier
pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA -v
if [ $? -eq 0 ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
exit 0
'''
