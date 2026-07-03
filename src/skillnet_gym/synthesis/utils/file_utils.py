"""File utility functions for Harbor synthesis pipeline"""

import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Literal


# File type mappings
FILE_TYPE_MAPPING = {
    # Documents
    "pdf": "document",
    "doc": "document",
    "docx": "document",
    "txt": "text",
    "md": "text",
    "rst": "text",

    # Spreadsheets
    "xlsx": "spreadsheet",
    "xls": "spreadsheet",
    "csv": "data",

    # Data formats
    "json": "data",
    "xml": "data",
    "yaml": "data",
    "yml": "data",
    "toml": "data",

    # Code files
    "py": "code",
    "js": "code",
    "ts": "code",
    "java": "code",
    "cpp": "code",
    "c": "code",
    "h": "code",
    "hpp": "code",
    "go": "code",
    "rs": "code",
    "rb": "code",
    "php": "code",
    "sh": "code",
    "bash": "code",
    "sql": "code",
    "html": "code",
    "css": "code",

    # Images
    "png": "image",
    "jpg": "image",
    "jpeg": "image",
    "gif": "image",
    "svg": "image",
    "webp": "image",

    # Archives
    "zip": "archive",
    "tar": "archive",
    "gz": "archive",
    "rar": "archive",

    # 3D/Binary
    "stl": "binary",
    "obj": "binary",
    "bin": "binary",
}


def get_file_extension(file_path: str) -> str:
    """Get lowercase file extension without the dot"""
    path = Path(file_path)
    ext = path.suffix.lower()
    return ext[1:] if ext.startswith(".") else ext


def detect_file_type(file_path: str) -> str:
    """
    Detect the type of a file based on its extension.

    Returns one of: document, text, spreadsheet, data, code, image, archive, binary, unknown
    """
    ext = get_file_extension(file_path)
    return FILE_TYPE_MAPPING.get(ext, "unknown")


def ensure_directory(path: str | Path) -> Path:
    """Ensure a directory exists, creating it if necessary"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_file(src: str | Path, dst: str | Path) -> Path:
    """
    Copy a file to destination.

    Args:
        src: Source file path
        dst: Destination path (file or directory)

    Returns:
        Path to the copied file
    """
    src = Path(src)
    dst = Path(dst)

    if dst.is_dir():
        dst = dst / src.name

    ensure_directory(dst.parent)
    shutil.copy2(src, dst)
    return dst


def copy_directory(src: str | Path, dst: str | Path, ignore_patterns: list[str] | None = None) -> Path:
    """
    Copy a directory recursively.

    Args:
        src: Source directory path
        dst: Destination directory path
        ignore_patterns: Patterns to ignore (e.g., ["*.pyc", "__pycache__"])

    Returns:
        Path to the copied directory
    """
    src = Path(src).resolve()
    dst = Path(dst).resolve()

    if src == dst:
        return dst

    if ignore_patterns:
        ignore = shutil.ignore_patterns(*ignore_patterns)
    else:
        ignore = None

    if dst.exists():
        shutil.rmtree(dst)

    shutil.copytree(src, dst, ignore=ignore)
    return dst


def create_working_directory(
    base_dir: str | Path | None = None,
    prefix: str = "harbor_work_",
) -> Path:
    """
    Create a temporary working directory.

    Args:
        base_dir: Base directory for the working dir, uses system temp if None
        prefix: Prefix for the directory name

    Returns:
        Path to the created directory
    """
    if base_dir:
        base_dir = Path(base_dir)
        ensure_directory(base_dir)
        work_dir = base_dir / f"{prefix}{uuid.uuid4().hex[:8]}"
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir
    else:
        return Path(tempfile.mkdtemp(prefix=prefix))


def setup_execution_directory(
    input_file: str | Path | None,
    skills_dir: str | Path | None,
    output_dir: str = "/root/output",
    working_base: str | Path | None = None,
) -> tuple[Path, Path | None, Path]:
    """
    Set up a complete execution directory with input files and skills.

    Args:
        input_file: Path to input file (optional)
        skills_dir: Path to skills directory (optional)
        output_dir: Output directory path within working dir
        working_base: Base directory for working dirs

    Returns:
        Tuple of (working_dir, input_file_in_working_dir, output_dir_path)
    """
    # Create working directory
    work_dir = create_working_directory(base_dir=working_base)

    # Copy input file if provided
    input_path = None
    if input_file:
        input_file = Path(input_file)
        if input_file.exists():
            input_path = copy_file(input_file, work_dir)

    # Set up output directory
    output_path = work_dir / output_dir.lstrip("/")
    ensure_directory(output_path)

    # Copy skills to .claude/skills if provided
    if skills_dir:
        skills_dir = Path(skills_dir)
        if skills_dir.exists():
            claude_skills_dir = work_dir / ".claude" / "skills"
            copy_directory(skills_dir, claude_skills_dir, ignore_patterns=["__pycache__", "*.pyc"])

    return work_dir, input_path, output_path


def get_relative_path(file_path: str | Path, base_dir: str | Path) -> str:
    """Get path relative to base directory"""
    file_path = Path(file_path).resolve()
    base_dir = Path(base_dir).resolve()
    try:
        return str(file_path.relative_to(base_dir))
    except ValueError:
        return str(file_path)


def list_files_recursive(
    directory: str | Path,
    extensions: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> list[Path]:
    """
    List all files in a directory recursively.

    Args:
        directory: Directory to search
        extensions: Filter by extensions (e.g., [".py", ".json"])
        ignore_patterns: Directory names to ignore

    Returns:
        List of file paths
    """
    directory = Path(directory)
    ignore_patterns = ignore_patterns or ["__pycache__", ".git", ".claude_runtime"]

    files = []
    for item in directory.rglob("*"):
        if item.is_file():
            # Check if any parent is in ignore patterns
            if any(pattern in item.parts for pattern in ignore_patterns):
                continue

            # Filter by extension if specified
            if extensions:
                if item.suffix.lower() in extensions:
                    files.append(item)
            else:
                files.append(item)

    return sorted(files)


def read_file_content(file_path: str | Path, encoding: str = "utf-8") -> str | None:
    """
    Read file content safely.

    Args:
        file_path: Path to file
        encoding: File encoding

    Returns:
        File content or None if read fails
    """
    try:
        with open(file_path, "r", encoding=encoding) as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return None


def write_file_content(
    file_path: str | Path,
    content: str,
    encoding: str = "utf-8",
) -> bool:
    """
    Write content to file safely.

    Args:
        file_path: Path to file
        content: Content to write
        encoding: File encoding

    Returns:
        True if write succeeded
    """
    try:
        file_path = Path(file_path)
        ensure_directory(file_path.parent)
        with open(file_path, "w", encoding=encoding) as f:
            f.write(content)
        return True
    except OSError:
        return False


def cleanup_directory(directory: str | Path) -> bool:
    """
    Remove a directory and all its contents.

    Args:
        directory: Directory to remove

    Returns:
        True if removal succeeded
    """
    try:
        shutil.rmtree(directory)
        return True
    except OSError:
        return False


def move_file(src: str | Path, dst: str | Path) -> Path | None:
    """
    Move a file to destination.

    Args:
        src: Source file path
        dst: Destination path (file or directory)

    Returns:
        Path to the moved file, or None if source doesn't exist
    """
    src = Path(src)
    dst = Path(dst)

    if not src.exists():
        return None

    if dst.is_dir():
        dst = dst / src.name

    ensure_directory(dst.parent)
    shutil.move(str(src), str(dst))
    return dst


def setup_isolated_workspace(
    task_id: str,
    skills_dir: str | Path,
    input_files: list[str | Path],
    exploration_summary_path: str | Path | None = None,
    file_metadata_path: str | Path | None = None,
    workspace_root: str = "./workspaces",
) -> dict[str, Path | list[Path]]:
    """
    Create an isolated workspace for task execution.

    This ensures Claude Code only sees the necessary files and all outputs
    are contained within the workspace directory.

    Args:
        task_id: Task ID, used as directory name
        skills_dir: Path to skills directory
        input_files: List of input file paths to copy
        exploration_summary_path: Path to exploration summary (optional)
        file_metadata_path: Path to file metadata JSON (optional)
        workspace_root: Root directory for workspaces

    Returns:
        Dictionary containing paths:
        {
            "workspace": workspace root path,
            "skills": skills directory path (where skill subdirs are placed),
            "input": input files directory path,
            "context": context documents directory path,
            "output": output directory path,
            "harbor_task": final task output path,
            "input_files": list of copied input file paths,
        }
    """
    workspace = Path(workspace_root) / task_id

    # Create directory structure
    dirs: dict[str, Path | list[Path]] = {
        "workspace": workspace,
        "skills": workspace / ".claude" / "skills",
        "input": workspace / "input",
        "context": workspace / "context",
        "output": workspace / "output",
        "harbor_task": workspace / "harbor_task",
    }

    # Create all directories
    for key, path in dirs.items():
        if isinstance(path, Path):
            path.mkdir(parents=True, exist_ok=True)

    # 1. Copy skills directory contents (preserves skill subdirectory names like "pdf", "excel")
    if skills_dir and Path(skills_dir).exists():
        skills_path = dirs["skills"]
        if isinstance(skills_path, Path):
            copy_directory(
                skills_dir,
                skills_path,  # Copy directly to .claude/skills/, not nested
                ignore_patterns=["__pycache__", "*.pyc"],
            )

    # 2. Copy input files
    copied_input_files: list[Path] = []
    input_path = dirs["input"]
    if isinstance(input_path, Path):
        for f in input_files:
            f = Path(f)
            if f.exists():
                copied = copy_file(f, input_path)
                copied_input_files.append(copied)

    # 3. Copy context documents
    context_path = dirs["context"]
    if isinstance(context_path, Path):
        if exploration_summary_path and Path(exploration_summary_path).exists():
            copy_file(exploration_summary_path, context_path)

        if file_metadata_path and Path(file_metadata_path).exists():
            copy_file(file_metadata_path, context_path)

    dirs["input_files"] = copied_input_files
    return dirs


def setup_exploration_workspace(
    skills_dir: str | Path,
    input_files: list[str | Path],
    file_summaries: dict[str, str] | None = None,
    workspace_root: str = "/tmp/harbor_exploration",
) -> dict[str, Path | list[Path]]:
    """
    Create an isolated workspace for exploration execution.

    This ensures Claude Code only sees the necessary files and cannot
    access files outside the workspace during exploration.

    Args:
        skills_dir: Path to skills directory
        input_files: List of input file paths to copy
        file_summaries: Optional file summaries (saved as JSON for reference)
        workspace_root: Root directory for temporary workspaces

    Returns:
        Dictionary containing paths:
        {
            "workspace": workspace root path,
            "skills": skills directory path,
            "input": input files directory path,
            "output": output directory path,
            "input_files": list of copied input file paths,
            "input_file_mapping": dict mapping original path to workspace path,
        }
    """
    import secrets
    from datetime import datetime

    # Generate unique workspace name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_hex = secrets.token_hex(4)
    workspace_name = f"exploration_{timestamp}_{random_hex}"
    workspace = Path(workspace_root) / workspace_name

    # Create directory structure
    dirs: dict[str, Path | list[Path] | dict[str, str]] = {
        "workspace": workspace,
        "skills": workspace / ".claude" / "skills",
        "input": workspace / "input",
        "output": workspace / "output",
    }

    # Create all directories
    for key, path in dirs.items():
        if isinstance(path, Path):
            path.mkdir(parents=True, exist_ok=True)

    # 1. Copy skills directory
    if skills_dir and Path(skills_dir).exists():
        skills_path = dirs["skills"]
        if isinstance(skills_path, Path):
            copy_directory(
                skills_dir,
                skills_path,
                ignore_patterns=["__pycache__", "*.pyc"],
            )

    # 2. Copy input files and create mapping
    copied_input_files: list[Path] = []
    input_file_mapping: dict[str, str] = {}  # original_path -> workspace_path
    input_path = dirs["input"]
    if isinstance(input_path, Path):
        for f in input_files:
            f = Path(f)
            if f.exists():
                copied = copy_file(f, input_path)
                copied_input_files.append(copied)
                input_file_mapping[str(f.resolve())] = str(copied)

    # 3. Save file summaries if provided
    if file_summaries:
        summaries_file = workspace / "file_summaries.json"
        import json
        # Convert to use workspace paths
        workspace_summaries = {}
        for orig_path, summary in file_summaries.items():
            workspace_path = input_file_mapping.get(str(Path(orig_path).resolve()))
            if workspace_path:
                workspace_summaries[workspace_path] = summary
            else:
                workspace_summaries[orig_path] = summary
        with open(summaries_file, "w", encoding="utf-8") as fp:
            json.dump(workspace_summaries, fp, ensure_ascii=False, indent=2)

    dirs["input_files"] = copied_input_files
    dirs["input_file_mapping"] = input_file_mapping
    return dirs


def cleanup_exploration_workspace(workspace_path: str | Path) -> bool:
    """
    Clean up an exploration workspace.

    Args:
        workspace_path: Path to the workspace to clean up

    Returns:
        True if cleanup was successful
    """
    workspace = Path(workspace_path)
    if workspace.exists() and workspace.is_dir():
        try:
            shutil.rmtree(workspace)
            return True
        except Exception as e:
            print(f"[Warning] Failed to cleanup workspace {workspace}: {e}")
            return False
    return True


def copy_exploration_results(
    workspace_path: str | Path,
    target_dir: str | Path,
    result_patterns: list[str] | None = None,
) -> list[Path]:
    """
    Copy exploration results from workspace to target directory.

    Args:
        workspace_path: Path to the exploration workspace
        target_dir: Target directory to copy results to
        result_patterns: Patterns for files to copy (default: state, summary, checkpoints)

    Returns:
        List of copied file paths
    """
    workspace = Path(workspace_path)
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    if result_patterns is None:
        result_patterns = [
            "dag_exploration_state.json",
            "exploration_state.json",
            "exploration_summary.md",
            "checkpoint_*.md",
        ]

    copied_files: list[Path] = []

    # Check output directory first (Claude might write there)
    output_dir = workspace / "output"
    search_dirs = [workspace, output_dir] if output_dir.exists() else [workspace]

    for search_dir in search_dirs:
        for pattern in result_patterns:
            for f in search_dir.glob(pattern):
                if f.is_file():
                    dest = target / f.name
                    shutil.copy2(f, dest)
                    copied_files.append(dest)

    return list(set(copied_files))  # Remove duplicates
