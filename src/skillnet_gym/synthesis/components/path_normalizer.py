"""Path normalizer for solve.sh scripts.

This module normalizes file paths in solve.sh to ensure they work
correctly in the target container environment.
"""

import re
from pathlib import Path


class PathNormalizer:
    """Normalizes paths in solve.sh to match container environment.

    During oracle execution, paths may reference the isolated workspace
    (e.g., /root/harbor_workspaces/task_id/). However, in the final
    container environment, files are at /root/.

    This class detects and normalizes such path mismatches.
    """

    # Patterns matching workspace paths that need normalization
    WORKSPACE_PATTERNS = [
        r'/root/harbor_workspaces/[^/\s"\']+/',
        r'/root/\.harbor/workspaces/[^/\s"\']+/',
        r'/root/harbor_workspaces/[^/\s"\']+',
        r'/tmp/harbor_[^/\s"\']+/',
    ]

    # Subdirectories commonly found in workspace that map to /root/
    # All should be flattened to /root/ for skillsbench compatibility
    WORKSPACE_SUBDIRS = [
        "input/",
        "output/",
        "final_res/",
        "workspace/",
    ]

    def __init__(
        self,
        target_root: str = "/root",
        input_files: list[str] | None = None,
    ):
        """Initialize the normalizer.

        Args:
            target_root: Target root directory in container
            input_files: List of expected input file names/paths
        """
        self.target_root = target_root
        self.input_files = input_files or []
        self._input_filenames = {
            Path(f).name for f in self.input_files
        }

    def normalize_solve_sh(self, solve_sh: str) -> str:
        """Normalize all paths in solve.sh content.

        Args:
            solve_sh: Original solve.sh content

        Returns:
            Normalized solve.sh content
        """
        result = solve_sh

        # Step 1: Replace workspace paths
        for pattern in self.WORKSPACE_PATTERNS:
            result = re.sub(pattern, f"{self.target_root}/", result)

        # Step 2: Clean up double slashes (except in http://)
        result = re.sub(r'(?<!:)//+', '/', result)

        # Step 3: Normalize common subdirectory patterns
        for subdir in self.WORKSPACE_SUBDIRS:
            # /root/input/file.pdf -> /root/file.pdf
            result = re.sub(
                rf'{self.target_root}/{subdir}([^/\s"\']+)',
                rf'{self.target_root}/\1',
                result
            )

        # Step 4: Fix Python string paths
        result = self._normalize_python_paths(result)

        return result

    def _normalize_python_paths(self, content: str) -> str:
        """Normalize paths inside Python code blocks in solve.sh.

        Args:
            content: solve.sh content

        Returns:
            Content with normalized Python paths
        """
        # Find Python heredocs
        python_pattern = r"python3?\s*<<\s*['\"]?(\w+)['\"]?\s*(.*?)^\1"

        def normalize_python_block(match):
            delimiter = match.group(1)
            python_code = match.group(2)

            # Normalize paths in the Python code
            for pattern in self.WORKSPACE_PATTERNS:
                python_code = re.sub(pattern, f"{self.target_root}/", python_code)

            # Clean up double slashes
            python_code = re.sub(r'(?<!:)//+', '/', python_code)

            return f"python3 << '{delimiter}'{python_code}{delimiter}"

        try:
            content = re.sub(
                python_pattern,
                normalize_python_block,
                content,
                flags=re.MULTILINE | re.DOTALL
            )
        except Exception:
            pass

        return content

    def extract_paths(self, solve_sh: str) -> list[str]:
        """Extract all file paths referenced in solve.sh.

        Args:
            solve_sh: solve.sh content

        Returns:
            List of extracted paths
        """
        paths = set()

        # Pattern for paths in various contexts
        path_patterns = [
            r'["\'](/[^"\'<>\s]+)["\']',  # Quoted paths
            r'\s(/root/[^\s"\'<>]+)',     # Unquoted /root paths
            r'(/tmp/[^\s"\'<>]+)',        # /tmp paths
        ]

        for pattern in path_patterns:
            for match in re.finditer(pattern, solve_sh):
                path = match.group(1)
                # Filter out obvious non-paths
                if not path.endswith(":") and not "://" in path:
                    paths.add(path)

        return sorted(paths)

    def validate_paths(
        self,
        solve_sh: str,
        working_dir: str | None = None,
    ) -> tuple[bool, list[str]]:
        """Validate that all paths in solve.sh are correct.

        Args:
            solve_sh: solve.sh content
            working_dir: Working directory for validation (optional)

        Returns:
            Tuple of (all_valid, list_of_invalid_paths)
        """
        paths = self.extract_paths(solve_sh)
        invalid_paths = []

        for path in paths:
            # Check for workspace path patterns that shouldn't be there
            for pattern in self.WORKSPACE_PATTERNS:
                if re.match(pattern, path):
                    invalid_paths.append(f"Workspace path not normalized: {path}")
                    break

            # Check input file references
            filename = Path(path).name
            if filename in self._input_filenames:
                # Input file should be at /root/{filename}
                expected = f"{self.target_root}/{filename}"
                if path != expected and not path.endswith(f"/{filename}"):
                    invalid_paths.append(
                        f"Input file path mismatch: {path} (expected {expected})"
                    )

            # Optional: Check if file exists
            if working_dir:
                # Map path to working_dir
                local_path = Path(working_dir) / Path(path).name
                # We don't fail on non-existence since files may be created during execution

        return len(invalid_paths) == 0, invalid_paths

    def normalize_for_container(
        self,
        solve_sh: str,
        input_files: list[str],
        output_dir: str = "/root",
    ) -> str:
        """Comprehensive normalization for container deployment.

        Args:
            solve_sh: Original solve.sh content
            input_files: List of input file paths (original locations)
            output_dir: Target output directory

        Returns:
            Fully normalized solve.sh content
        """
        result = solve_sh

        # 1. Normalize workspace paths
        result = self.normalize_solve_sh(result)

        # 2. Normalize input file references
        for input_file in input_files:
            original_name = Path(input_file).name
            # Replace various forms of the path with /root/{filename}
            patterns_to_replace = [
                rf'["\']?{re.escape(input_file)}["\']?',
                rf'/root/input/{re.escape(original_name)}',
                rf'/root/workspace/{re.escape(original_name)}',
            ]
            target = f"{output_dir}/{original_name}"

            for pattern in patterns_to_replace:
                result = re.sub(pattern, f'"{target}"', result)

        # 3. Ensure output directory references are correct
        # Replace /root/output/ with /root/ (flat structure)
        result = re.sub(r'/root/output/([^/\s"\']+)', r'/root/\1', result)

        # 4. Final cleanup
        result = re.sub(r'(?<!:)//+', '/', result)

        return result


def create_path_normalizer(input_files: list[str]) -> PathNormalizer:
    """Factory function to create a PathNormalizer.

    Args:
        input_files: List of input file paths

    Returns:
        Configured PathNormalizer instance
    """
    return PathNormalizer(
        target_root="/root",
        input_files=input_files,
    )
