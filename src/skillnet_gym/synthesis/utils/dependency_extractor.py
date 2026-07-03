"""
Unified Dependency Extractor

Extracts runtime dependencies from multiple sources:
1. Skill directory analysis (requirements.txt, dependencies.json, SKILL.md)
2. Trajectory analysis (pip/apt/conda install commands, import statements)
3. Runtime capture (pip freeze after execution)

Consolidates and deduplicates dependencies for Dockerfile generation.
"""

import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Standard library modules (Python 3.10+) - should not be included in pip packages
STDLIB_MODULES = {
    "abc", "aifc", "argparse", "array", "ast", "asynchat", "asyncio", "asyncore",
    "atexit", "audioop", "base64", "bdb", "binascii", "binhex", "bisect",
    "builtins", "bz2", "calendar", "cgi", "cgitb", "chunk", "cmath", "cmd",
    "code", "codecs", "codeop", "collections", "colorsys", "compileall",
    "concurrent", "configparser", "contextlib", "contextvars", "copy", "copyreg",
    "cProfile", "crypt", "csv", "ctypes", "curses", "dataclasses", "datetime",
    "dbm", "decimal", "difflib", "dis", "distutils", "doctest", "email",
    "encodings", "enum", "errno", "faulthandler", "fcntl", "filecmp", "fileinput",
    "fnmatch", "fractions", "ftplib", "functools", "gc", "getopt", "getpass",
    "gettext", "glob", "graphlib", "grp", "gzip", "hashlib", "heapq", "hmac",
    "html", "http", "idlelib", "imaplib", "imghdr", "imp", "importlib", "inspect",
    "io", "ipaddress", "itertools", "json", "keyword", "lib2to3", "linecache",
    "locale", "logging", "lzma", "mailbox", "mailcap", "marshal", "math",
    "mimetypes", "mmap", "modulefinder", "multiprocessing", "netrc", "nis",
    "nntplib", "numbers", "operator", "optparse", "os", "ossaudiodev", "pathlib",
    "pdb", "pickle", "pickletools", "pipes", "pkgutil", "platform", "plistlib",
    "poplib", "posix", "posixpath", "pprint", "profile", "pstats", "pty", "pwd",
    "py_compile", "pyclbr", "pydoc", "queue", "quopri", "random", "re", "readline",
    "reprlib", "resource", "rlcompleter", "runpy", "sched", "secrets", "select",
    "selectors", "shelve", "shlex", "shutil", "signal", "site", "smtpd", "smtplib",
    "sndhdr", "socket", "socketserver", "spwd", "sqlite3", "ssl", "stat",
    "statistics", "string", "stringprep", "struct", "subprocess", "sunau",
    "symtable", "sys", "sysconfig", "syslog", "tabnanny", "tarfile", "telnetlib",
    "tempfile", "termios", "test", "textwrap", "threading", "time", "timeit",
    "tkinter", "token", "tokenize", "trace", "traceback", "tracemalloc", "tty",
    "turtle", "turtledemo", "types", "typing", "unicodedata", "unittest", "urllib",
    "uu", "uuid", "venv", "warnings", "wave", "weakref", "webbrowser", "winreg",
    "winsound", "wsgiref", "xdrlib", "xml", "xmlrpc", "zipapp", "zipfile",
    "zipimport", "zlib", "_thread", "__future__",
}

# Common package name mappings (import name -> pip package name)
IMPORT_TO_PACKAGE = {
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
    "dotenv": "python-dotenv",
    "dateutil": "python-dateutil",
    "wx": "wxPython",
    "serial": "pyserial",
    "usb": "pyusb",
    "Bio": "biopython",
    "rdkit": "rdkit",
    "cv": "opencv-python",
    "docx": "python-docx",
    "pptx": "python-pptx",
    "xlrd": "xlrd",
    "xlwt": "xlwt",
    "openpyxl": "openpyxl",
    "fitz": "PyMuPDF",
    "PyPDF2": "PyPDF2",
    "pdfplumber": "pdfplumber",
    "tabula": "tabula-py",
    "camelot": "camelot-py",
    "pdf2image": "pdf2image",
    "pytesseract": "pytesseract",
    "pyperclip": "pyperclip",
    "requests": "requests",
    "httpx": "httpx",
    "aiohttp": "aiohttp",
    "flask": "Flask",
    "django": "Django",
    "fastapi": "fastapi",
    "sqlalchemy": "SQLAlchemy",
    "psycopg2": "psycopg2-binary",
    "pymongo": "pymongo",
    "redis": "redis",
    "celery": "celery",
    "boto3": "boto3",
    "google": "google-cloud",
    "azure": "azure",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "plotly": "plotly",
    "torch": "torch",
    "tensorflow": "tensorflow",
    "keras": "keras",
    "transformers": "transformers",
    "networkx": "networkx",
    "sympy": "sympy",
    "nltk": "nltk",
    "spacy": "spacy",
    "gensim": "gensim",
    "jieba": "jieba",
    "langchain": "langchain",
    "openai": "openai",
    "anthropic": "anthropic",
}

# System dependencies patterns (keyword -> apt package)
SYSTEM_DEP_PATTERNS = {
    r"poppler|pdftotext|pdftoppm|pdfimages": ["poppler-utils"],
    r"tesseract|pytesseract": ["tesseract-ocr", "tesseract-ocr-chi-sim"],
    r"ffmpeg": ["ffmpeg"],
    r"ghostscript|gs\s": ["ghostscript"],
    r"imagemagick|convert\s|magick": ["imagemagick"],
    r"pandoc": ["pandoc"],
    r"graphviz|dot\s": ["graphviz"],
    r"libgl|opencv|cv2": ["libgl1-mesa-glx", "libglib2.0-0"],
    r"wkhtmlto": ["wkhtmltopdf"],
    r"chromium|chrome|puppeteer": ["chromium-browser"],
    r"libreoffice|soffice": ["libreoffice"],
}


@dataclass
class ExtractedDependencies:
    """Container for extracted dependencies from all sources"""
    pip_packages: set = field(default_factory=set)
    apt_packages: set = field(default_factory=set)
    conda_packages: set = field(default_factory=set)

    # Track sources for debugging
    sources: dict = field(default_factory=lambda: {
        "skill_requirements": [],
        "skill_dependencies_json": [],
        "skill_md_analysis": [],
        "trajectory_commands": [],
        "trajectory_imports": [],
        "runtime_freeze": [],
    })

    def merge_from(self, other: "ExtractedDependencies") -> None:
        """Merge dependencies from another instance"""
        self.pip_packages.update(other.pip_packages)
        self.apt_packages.update(other.apt_packages)
        self.conda_packages.update(other.conda_packages)
        for key in self.sources:
            if key in other.sources:
                self.sources[key].extend(other.sources[key])

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary"""
        return {
            "pip_packages": sorted(self.pip_packages),
            "apt_packages": sorted(self.apt_packages),
            "conda_packages": sorted(self.conda_packages),
            "sources": {k: list(set(v)) for k, v in self.sources.items()},
        }

    def get_pip_install_command(self) -> str:
        """Generate pip install command"""
        if not self.pip_packages:
            return ""
        packages = " ".join(sorted(self.pip_packages))
        return f"pip3 install --break-system-packages {packages}"

    def get_apt_install_command(self) -> str:
        """Generate apt-get install command"""
        if not self.apt_packages:
            return ""
        packages = " ".join(sorted(self.apt_packages))
        return f"apt-get install -y {packages}"

    def get_conda_install_command(self) -> str:
        """Generate conda install command"""
        if not self.conda_packages:
            return ""
        packages = " ".join(sorted(self.conda_packages))
        return f"conda install -y {packages}"


class DependencyExtractor:
    """
    Unified dependency extractor that combines multiple extraction methods.

    Usage:
        extractor = DependencyExtractor()

        # Extract from skill directory
        deps = extractor.extract_from_skill_dir("/path/to/skill")

        # Extract from trajectory
        deps.merge_from(extractor.extract_from_trajectory(trajectory))

        # Extract from runtime (optional)
        deps.merge_from(extractor.extract_from_runtime(working_dir))

        # Generate Dockerfile commands
        print(deps.get_apt_install_command())
        print(deps.get_pip_install_command())
    """

    def __init__(self):
        self.stdlib_modules = STDLIB_MODULES
        self.import_to_package = IMPORT_TO_PACKAGE
        self.system_dep_patterns = SYSTEM_DEP_PATTERNS

    # =========================================================================
    # 1. Skill Directory Analysis
    # =========================================================================

    def extract_from_skill_dir(self, skill_dir: str | Path) -> ExtractedDependencies:
        """
        Extract dependencies from a skill directory.

        Checks:
        1. requirements.txt - pip packages
        2. dependencies.json - structured dependencies (pip/apt/conda)
        3. SKILL.md - system dependencies via keyword matching

        Args:
            skill_dir: Path to skill directory

        Returns:
            ExtractedDependencies instance
        """
        deps = ExtractedDependencies()
        skill_path = Path(skill_dir)

        if not skill_path.exists():
            return deps

        # 1. Parse requirements.txt
        requirements_file = skill_path / "requirements.txt"
        if requirements_file.exists():
            packages = self._parse_requirements_txt(requirements_file)
            deps.pip_packages.update(packages)
            deps.sources["skill_requirements"].extend(packages)

        # 2. Parse dependencies.json (new format)
        deps_file = skill_path / "dependencies.json"
        if deps_file.exists():
            json_deps = self._parse_dependencies_json(deps_file)
            deps.pip_packages.update(json_deps.get("pip", []))
            deps.apt_packages.update(json_deps.get("apt", []))
            deps.conda_packages.update(json_deps.get("conda", []))
            deps.sources["skill_dependencies_json"].extend(json_deps.get("pip", []))

        # 3. Scan SKILL.md for system dependencies
        skill_md = skill_path / "SKILL.md"
        if skill_md.exists():
            system_deps = self._extract_system_deps_from_md(skill_md)
            deps.apt_packages.update(system_deps)
            deps.sources["skill_md_analysis"].extend(system_deps)

        # 4. Scan Python scripts in scripts/ directory for imports
        scripts_dir = skill_path / "scripts"
        if scripts_dir.exists():
            for script_file in scripts_dir.glob("*.py"):
                imports = self._extract_imports_from_file(script_file)
                packages = self._convert_imports_to_packages(imports)
                deps.pip_packages.update(packages)
                deps.sources["skill_requirements"].extend(packages)

        return deps

    def extract_from_skills_dir(self, skills_dir: str | Path) -> ExtractedDependencies:
        """
        Extract dependencies from all skills in a directory.

        Args:
            skills_dir: Path to directory containing skill subdirectories

        Returns:
            Merged ExtractedDependencies from all skills
        """
        deps = ExtractedDependencies()
        skills_path = Path(skills_dir)

        if not skills_path.exists():
            return deps

        for skill_folder in skills_path.iterdir():
            if skill_folder.is_dir() and not skill_folder.name.startswith("."):
                skill_deps = self.extract_from_skill_dir(skill_folder)
                deps.merge_from(skill_deps)

        return deps

    def _parse_requirements_txt(self, file_path: Path) -> list[str]:
        """Parse requirements.txt and extract package names"""
        packages = []
        try:
            content = file_path.read_text(encoding="utf-8")
            for line in content.splitlines():
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue
                # Skip -r includes and other options
                if line.startswith("-"):
                    continue
                # Extract package name (handle version specifiers)
                match = re.match(r"^([a-zA-Z0-9_\-\.]+)(\[.*\])?", line)
                if match:
                    packages.append(match.group(0))
        except Exception:
            pass
        return packages

    def _parse_dependencies_json(self, file_path: Path) -> dict[str, list[str]]:
        """
        Parse dependencies.json file.

        Expected format:
        {
            "pip": ["package1", "package2>=1.0"],
            "apt": ["libfoo-dev", "libbar"],
            "conda": ["rdkit"],
            "python": ">=3.10"
        }
        """
        try:
            content = file_path.read_text(encoding="utf-8")
            data = json.loads(content)
            return {
                "pip": data.get("pip", []),
                "apt": data.get("apt", []),
                "conda": data.get("conda", []),
            }
        except Exception:
            return {"pip": [], "apt": [], "conda": []}

    def _extract_system_deps_from_md(self, skill_md: Path) -> list[str]:
        """Extract system dependencies from SKILL.md via keyword matching"""
        system_deps = []
        try:
            content = skill_md.read_text(encoding="utf-8").lower()
            for pattern, apt_packages in self.system_dep_patterns.items():
                if re.search(pattern, content):
                    system_deps.extend(apt_packages)
        except Exception:
            pass
        return list(set(system_deps))

    # =========================================================================
    # 2. Trajectory Analysis
    # =========================================================================

    def extract_from_trajectory(self, trajectory: Any) -> ExtractedDependencies:
        """
        Extract dependencies from execution trajectory.

        Analyzes:
        1. Bash commands for pip/apt/conda install
        2. Written Python files for import statements

        Args:
            trajectory: Trajectory object with steps

        Returns:
            ExtractedDependencies instance
        """
        deps = ExtractedDependencies()

        if not hasattr(trajectory, "steps"):
            return deps

        for step in trajectory.steps:
            if step.action_type != "tool_use" or not step.tool_input:
                continue

            if step.tool_name == "Bash":
                command = step.tool_input.get("command", "")
                cmd_deps = self._extract_from_bash_command(command)
                deps.merge_from(cmd_deps)

            elif step.tool_name == "Write":
                file_path = step.tool_input.get("file_path", "")
                content = step.tool_input.get("content", "")
                if file_path.endswith(".py") and content:
                    imports = self._extract_imports_from_code(content)
                    packages = self._convert_imports_to_packages(imports)
                    deps.pip_packages.update(packages)
                    deps.sources["trajectory_imports"].extend(packages)

        return deps

    def _extract_from_bash_command(self, command: str) -> ExtractedDependencies:
        """Extract dependencies from a bash command"""
        deps = ExtractedDependencies()

        # 1. pip install patterns
        pip_patterns = [
            r'pip3?\s+install\s+(?:--[^\s]+\s+)*([^\s&|;#]+)',
            r'python3?\s+-m\s+pip\s+install\s+(?:--[^\s]+\s+)*([^\s&|;#]+)',
        ]
        for pattern in pip_patterns:
            matches = re.findall(pattern, command)
            for match in matches:
                if not match.startswith("-") and match:
                    # Clean version specifiers for tracking
                    pkg_name = re.split(r"[=<>!]", match)[0]
                    if pkg_name:
                        deps.pip_packages.add(match)  # Keep version specifier
                        deps.sources["trajectory_commands"].append(match)

        # 2. apt-get install patterns
        apt_patterns = [
            r'apt-get\s+install\s+(?:-y\s+)?([^\s&|;#]+)',
            r'apt\s+install\s+(?:-y\s+)?([^\s&|;#]+)',
        ]
        for pattern in apt_patterns:
            matches = re.findall(pattern, command)
            for match in matches:
                if not match.startswith("-") and match:
                    deps.apt_packages.add(match)
                    deps.sources["trajectory_commands"].append(f"apt:{match}")

        # 3. conda install patterns
        conda_patterns = [
            r'conda\s+install\s+(?:-y\s+)?(?:-c\s+\w+\s+)?([^\s&|;#]+)',
            r'mamba\s+install\s+(?:-y\s+)?(?:-c\s+\w+\s+)?([^\s&|;#]+)',
        ]
        for pattern in conda_patterns:
            matches = re.findall(pattern, command)
            for match in matches:
                if not match.startswith("-") and match:
                    deps.conda_packages.add(match)
                    deps.sources["trajectory_commands"].append(f"conda:{match}")

        return deps

    def _extract_imports_from_file(self, file_path: Path) -> set[str]:
        """Extract import statements from a Python file"""
        try:
            content = file_path.read_text(encoding="utf-8")
            return self._extract_imports_from_code(content)
        except Exception:
            return set()

    def _extract_imports_from_code(self, code: str) -> set[str]:
        """Extract import statements from Python code string"""
        imports = set()

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        # Get top-level module name
                        module = alias.name.split(".")[0]
                        imports.add(module)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module = node.module.split(".")[0]
                        imports.add(module)
        except SyntaxError:
            # Fallback to regex for invalid Python
            import_pattern = r'^(?:from\s+(\w+)|import\s+(\w+))'
            for line in code.splitlines():
                match = re.match(import_pattern, line.strip())
                if match:
                    module = match.group(1) or match.group(2)
                    if module:
                        imports.add(module)

        return imports

    def _convert_imports_to_packages(self, imports: set[str]) -> list[str]:
        """Convert import names to pip package names"""
        packages = []
        for imp in imports:
            # Skip standard library
            if imp in self.stdlib_modules:
                continue
            # Map known import names to package names
            if imp in self.import_to_package:
                packages.append(self.import_to_package[imp])
            else:
                # Assume import name == package name (common case)
                packages.append(imp)
        return packages

    # =========================================================================
    # 3. Runtime Capture
    # =========================================================================

    def extract_from_runtime(
        self,
        working_dir: str | None = None,
        baseline_packages: set[str] | None = None,
    ) -> ExtractedDependencies:
        """
        Capture installed packages from runtime environment.

        Useful for detecting packages installed during execution
        that weren't captured through command analysis.

        Args:
            working_dir: Working directory for pip command
            baseline_packages: Pre-execution packages to exclude (optional)

        Returns:
            ExtractedDependencies with pip packages from freeze
        """
        deps = ExtractedDependencies()

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True,
                text=True,
                cwd=working_dir,
                timeout=30,
            )

            if result.returncode == 0:
                for line in result.stdout.strip().splitlines():
                    if "==" in line:
                        pkg = line.split("==")[0].strip()
                        # Skip if in baseline
                        if baseline_packages and pkg.lower() in baseline_packages:
                            continue
                        deps.pip_packages.add(pkg)
                        deps.sources["runtime_freeze"].append(pkg)
        except Exception:
            pass

        return deps

    def get_baseline_packages(self) -> set[str]:
        """Get current environment's packages as baseline"""
        packages = set()
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().splitlines():
                    if "==" in line:
                        packages.add(line.split("==")[0].strip().lower())
        except Exception:
            pass
        return packages

    # =========================================================================
    # 4. Combined Extraction
    # =========================================================================

    def extract_all(
        self,
        skill_dir: str | Path | None = None,
        skills_dir: str | Path | None = None,
        trajectory: Any = None,
        working_dir: str | None = None,
        capture_runtime: bool = False,
    ) -> ExtractedDependencies:
        """
        Extract dependencies from all available sources.

        Args:
            skill_dir: Single skill directory path
            skills_dir: Directory containing multiple skills
            trajectory: Execution trajectory
            working_dir: Working directory for runtime capture
            capture_runtime: Whether to capture runtime packages

        Returns:
            Merged ExtractedDependencies from all sources
        """
        deps = ExtractedDependencies()

        # 1. Extract from skill directory
        if skill_dir:
            deps.merge_from(self.extract_from_skill_dir(skill_dir))

        # 2. Extract from skills directory (multiple skills)
        if skills_dir:
            deps.merge_from(self.extract_from_skills_dir(skills_dir))

        # 3. Extract from trajectory
        if trajectory:
            deps.merge_from(self.extract_from_trajectory(trajectory))

        # 4. Capture runtime packages (optional)
        if capture_runtime and working_dir:
            deps.merge_from(self.extract_from_runtime(working_dir))

        # Post-process: deduplicate and clean
        self._cleanup_dependencies(deps)

        return deps

    def _cleanup_dependencies(self, deps: ExtractedDependencies) -> None:
        """Clean up and deduplicate dependencies"""
        # Remove duplicates with different version specifiers (keep versioned)
        pip_cleaned = {}
        for pkg in deps.pip_packages:
            base_name = re.split(r"[=<>!\[]", pkg)[0].lower()
            existing = pip_cleaned.get(base_name)
            if existing is None:
                pip_cleaned[base_name] = pkg
            elif "==" in pkg or ">=" in pkg:
                # Prefer versioned over unversioned
                pip_cleaned[base_name] = pkg

        deps.pip_packages = set(pip_cleaned.values())

        # Remove packages that are clearly not pip packages
        invalid_patterns = [
            r"^-",  # Flags
            r"^\.",  # Relative paths
            r"^/",  # Absolute paths
            r"^\d",  # Starting with number
        ]
        deps.pip_packages = {
            p for p in deps.pip_packages
            if not any(re.match(pat, p) for pat in invalid_patterns)
        }


def create_dependencies_json_template(skill_dir: str | Path) -> str:
    """
    Create a dependencies.json template for a skill directory.

    Analyzes existing requirements.txt and SKILL.md to pre-populate.

    Args:
        skill_dir: Path to skill directory

    Returns:
        JSON string template
    """
    extractor = DependencyExtractor()
    deps = extractor.extract_from_skill_dir(skill_dir)

    template = {
        "pip": sorted(deps.pip_packages),
        "apt": sorted(deps.apt_packages),
        "conda": sorted(deps.conda_packages),
        "python": ">=3.10",
    }

    return json.dumps(template, indent=2, ensure_ascii=False)
