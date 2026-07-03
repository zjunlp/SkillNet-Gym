"""Path normalization for Harbor task synthesis.

This module provides utilities to normalize file paths in synthesized artifacts,
ensuring all paths use the correct container paths (/root/) instead of local
temporary workspace paths.
"""

import re
from pathlib import Path
from typing import Callable


class PathNormalizer:
    """Normalizes paths in synthesized artifacts to container paths.

    During synthesis, files are processed in temporary workspaces like:
    - /tmp/harbor_exploration/exploration_xxx/input/file.pdf
    - /root/harbor_workspaces/task_xxx/input/file.pdf
    - <host_share>/.../file.pdf  (e.g. NFS-mounted dataset shares)

    This class normalizes all such paths to container paths like:
    - /root/file.pdf (for input files)
    - /root/output/result.json (for output files)

    Note: the regexes below also strip a few host-specific dataset-share
    prefixes (``/ossfs/workspace/...``, ``/dtai/...``) that were observed
    in the trajectories used to develop this pipeline. They are harmless
    for other setups; add more patterns here if your infra uses different
    host paths.
    """

    def __init__(
        self,
        input_files: list[str],
        container_workdir: str = "/root",
    ):
        """
        Initialize the path normalizer.

        Args:
            input_files: Original input file paths (will be mapped to container paths)
            container_workdir: Container working directory (default: /root)
        """
        self.workdir = container_workdir
        self.path_mapping: dict[str, str] = {}
        self._filenames: set[str] = set()

        # Build path mapping for input files
        for f in input_files:
            if not f:
                continue
            filename = Path(f).name
            container_path = f"{self.workdir}/{filename}"

            # Map various forms of the original path
            self.path_mapping[f] = container_path
            try:
                resolved = str(Path(f).resolve())
                self.path_mapping[resolved] = container_path
            except Exception:
                pass

            self._filenames.add(filename)

    def normalize(self, text: str) -> str:
        """Normalize all paths in text to container paths.

        Args:
            text: Text content (instruction, test code, shell script, etc.)

        Returns:
            Text with all paths normalized to container paths
        """
        if not text:
            return text

        result = text

        # 1. Replace known full paths (longer paths first to avoid partial matches)
        for orig_path, container_path in sorted(
            self.path_mapping.items(),
            key=lambda x: -len(x[0])
        ):
            result = result.replace(orig_path, container_path)

        # 2. Replace various temporary workspace path patterns
        patterns: list[tuple[str, Callable[[re.Match], str]]] = [
            # /tmp/harbor_exploration/.../input/filename
            (r'/tmp/harbor_exploration/[^\s"\'`\]\)\n]+/input/([^\s"\'`\]\)\n]+)',
             self._get_container_path_for_input),

            # /tmp/harbor_exploration/.../output/filename
            (r'/tmp/harbor_exploration/[^\s"\'`\]\)\n]+/output/([^\s"\'`\]\)\n]+)',
             self._get_container_path_for_output),

            # /root/harbor_workspaces/.../input/filename
            (r'/root/harbor_workspaces/[^\s"\'`\]\)\n]+/input/([^\s"\'`\]\)\n]+)',
             self._get_container_path_for_input),

            # /root/harbor_workspaces/.../output/filename
            (r'/root/harbor_workspaces/[^\s"\'`\]\)\n]+/output/([^\s"\'`\]\)\n]+)',
             self._get_container_path_for_output),

            # /dtai/share/nlp/.../filename (user-reported problematic paths)
            (r'/dtai/[^\s"\'`\]\)\n]+/([^\s"\'`\]\)/\n]+\.[a-zA-Z0-9]+)',
             self._get_container_path_for_input),

            # /ossfs/workspace/.../input/filename
            (r'/ossfs/workspace/[^\s"\'`\]\)\n]+/input/([^\s"\'`\]\)\n]+)',
             self._get_container_path_for_input),

            # /ossfs/workspace/.../output/filename
            (r'/ossfs/workspace/[^\s"\'`\]\)\n]+/output/([^\s"\'`\]\)\n]+)',
             self._get_container_path_for_output),

            # Generic .../input/filename pattern (catch-all, be more careful)
            (r'(?<![a-zA-Z0-9])/[^\s"\'`\[\(\n]+/input/([^\s"\'`\]\)\n]+\.[a-zA-Z0-9]+)',
             self._get_container_path_for_input),
        ]

        for pattern, replacer in patterns:
            result = re.sub(pattern, replacer, result)

        # 3. Normalize /root/output/ paths
        # Replace /root/output/xxx with /root/xxx for known input files
        for filename in self._filenames:
            result = result.replace(f'/root/output/{filename}', f'{self.workdir}/{filename}')

        return result

    def _get_container_path_for_input(self, match: re.Match) -> str:
        """Get container path for a matched input filename."""
        filename = match.group(1)
        if filename in self._filenames:
            return f"{self.workdir}/{filename}"
        # Unknown file - could be input or output, default to workdir
        return f"{self.workdir}/{filename}"

    def _get_container_path_for_output(self, match: re.Match) -> str:
        """Get container path for a matched output filename."""
        filename = match.group(1)
        if filename in self._filenames:
            # This is an input file referenced as output, use workdir
            return f"{self.workdir}/{filename}"
        # Output file
        return f"{self.workdir}/{filename}"

    def normalize_instruction(self, instruction: str) -> str:
        """Normalize paths in instruction.md content.

        Args:
            instruction: Instruction markdown content

        Returns:
            Instruction with normalized paths
        """
        return self.normalize(instruction)

    def normalize_tests(self, tests: str) -> str:
        """Normalize paths in test file content.

        Args:
            tests: Test Python code

        Returns:
            Test code with normalized paths
        """
        return self.normalize(tests)

    def normalize_solve_sh(self, solve_sh: str) -> str:
        """Normalize paths in solve.sh content.

        Args:
            solve_sh: Shell script content

        Returns:
            Shell script with normalized paths
        """
        return self.normalize(solve_sh)


def normalize_paths_in_text(
    text: str,
    input_files: list[str],
    container_workdir: str = "/root",
) -> str:
    """Convenience function to normalize paths in text.

    Args:
        text: Text content to normalize
        input_files: List of original input file paths
        container_workdir: Container working directory

    Returns:
        Text with normalized paths
    """
    normalizer = PathNormalizer(input_files, container_workdir)
    return normalizer.normalize(text)


def normalize_for_skillsbench(
    content: str,
    workspace_path: str | None = None,
    container_root: str = "/root",
) -> str:
    """Normalize paths in content for skillsbench container format.

    This function converts workspace paths to container paths suitable for
    skillsbench execution environments.

    Examples:
        /root/harbor_workspaces/task_001/input/data.csv -> /root/data.csv
        /root/harbor_workspaces/task_001/final_res/output.json -> /root/output.json
        /tmp/solve_verify_xxx/data.csv -> /root/data.csv

    Args:
        content: Text content (test code, solve.sh, etc.)
        workspace_path: Specific workspace path to replace (if known)
        container_root: Container root path (default: /root)

    Returns:
        Content with normalized container paths
    """
    if not content:
        return content

    result = content

    # 1. Replace specific workspace path if provided
    if workspace_path:
        # Normalize workspace path (ensure no trailing slash)
        workspace_path = workspace_path.rstrip("/")
        result = result.replace(workspace_path + "/", container_root + "/")
        result = result.replace(workspace_path, container_root)

    # 2. Replace common workspace path patterns
    patterns = [
        # /root/harbor_workspaces/<task>/input/<file>
        (r'/root/harbor_workspaces/[^/]+/input/', f'{container_root}/'),
        # /root/harbor_workspaces/<task>/final_res/<file>
        (r'/root/harbor_workspaces/[^/]+/final_res/', f'{container_root}/'),
        # /root/harbor_workspaces/<task>/output/<file>
        (r'/root/harbor_workspaces/[^/]+/output/', f'{container_root}/'),
        # /root/harbor_workspaces/<task>/<file> (direct)
        (r'/root/harbor_workspaces/[^/]+/', f'{container_root}/'),
        # /tmp/harbor_exploration/<id>/input/<file>
        (r'/tmp/harbor_exploration/[^/]+/input/', f'{container_root}/'),
        # /tmp/harbor_exploration/<id>/output/<file>
        (r'/tmp/harbor_exploration/[^/]+/output/', f'{container_root}/'),
        # /tmp/harbor_exploration/<id>/<file> (direct)
        (r'/tmp/harbor_exploration/[^/]+/', f'{container_root}/'),
        # /tmp/solve_verify_<id>/<file>
        (r'/tmp/solve_verify_[^/]+/', f'{container_root}/'),
        # /ossfs/workspace/.../input/<file>
        (r'/ossfs/workspace/[^\s"\'`\]\)\n]+/input/', f'{container_root}/'),
        # /ossfs/workspace/.../output/<file>
        (r'/ossfs/workspace/[^\s"\'`\]\)\n]+/output/', f'{container_root}/'),
    ]

    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result)

    # 3. Remove /input/, /output/, /final_res/ subdirectories (skillsbench uses flat /root/ structure)
    # These patterns handle cases where workspace path was already replaced but subdirs remain
    flat_patterns = [
        (r'/root/input/', f'{container_root}/'),
        (r'/root/output/', f'{container_root}/'),
        (r'/root/final_res/', f'{container_root}/'),
    ]
    for pattern, replacement in flat_patterns:
        result = result.replace(pattern, replacement)

    # 4. Clean up double slashes
    result = re.sub(r'/root/+', '/root/', result)

    return result


class SkillsbenchPathNormalizer:
    """Path normalizer specifically for skillsbench format output.

    This class provides stateful path normalization for converting workspace
    paths to container paths, suitable for pytest and solve.sh generation.
    """

    def __init__(
        self,
        workspace_path: str | None = None,
        input_files: list[str] | None = None,
        container_root: str = "/root",
    ):
        """
        Initialize the skillsbench path normalizer.

        Args:
            workspace_path: Current workspace path (for direct replacement)
            input_files: List of input file paths (for filename tracking)
            container_root: Container root path (default: /root)
        """
        self.workspace_path = workspace_path.rstrip("/") if workspace_path else None
        self.container_root = container_root
        self.input_filenames: set[str] = set()

        # Extract input filenames
        if input_files:
            for f in input_files:
                if f:
                    self.input_filenames.add(Path(f).name)

    def normalize(self, content: str) -> str:
        """Normalize all paths in content for skillsbench.

        Args:
            content: Text content to normalize

        Returns:
            Content with normalized paths
        """
        return normalize_for_skillsbench(
            content=content,
            workspace_path=self.workspace_path,
            container_root=self.container_root,
        )

    def normalize_tests(self, tests_content: str) -> str:
        """Normalize paths in pytest test content.

        Args:
            tests_content: Python test code

        Returns:
            Test code with normalized paths
        """
        return self.normalize(tests_content)

    def normalize_solve_sh(self, solve_sh_content: str) -> str:
        """Normalize paths in solve.sh content.

        Args:
            solve_sh_content: Shell script content

        Returns:
            Shell script with normalized paths
        """
        return self.normalize(solve_sh_content)
