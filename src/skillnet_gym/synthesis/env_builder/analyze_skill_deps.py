"""
Skill Dependency Analyzer

Analyzes skill folders to extract pip and system dependencies.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


# Known system dependencies that appear in SKILL.md files
# Maps keywords to conda-forge package names
SYSTEM_DEP_PATTERNS = {
    r"poppler|pdftotext|pdftoppm|pdfimages": "poppler",
    r"tesseract|pytesseract": "tesseract",
    r"ffmpeg": "ffmpeg",
    r"ghostscript|gs\s": "ghostscript",
    r"imagemagick|convert\s": "imagemagick",
    r"pandoc": "pandoc",
    r"graphviz|dot\s": "graphviz",
}


@dataclass
class SkillDependency:
    """Represents dependencies for a single skill"""
    name: str
    pip_packages: list[str] = field(default_factory=list)
    conda_packages: list[str] = field(default_factory=list)
    system_deps: list[str] = field(default_factory=list)
    skill_path: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "pip_packages": self.pip_packages,
            "conda_packages": self.conda_packages,
            "system_deps": self.system_deps,
            "skill_path": self.skill_path,
        }


class SkillDependencyAnalyzer:
    """Analyzes skill directories to extract dependencies"""

    def __init__(self, skills_dir: str):
        """
        Initialize analyzer.

        Args:
            skills_dir: Path to the Skill-pair directory
        """
        self.skills_dir = Path(skills_dir)
        self.skills: dict[str, SkillDependency] = {}

    def analyze_all(self) -> dict[str, SkillDependency]:
        """
        Analyze all skills in the directory.

        Returns:
            Dictionary mapping skill name to SkillDependency
        """
        if not self.skills_dir.exists():
            raise FileNotFoundError(f"Skills directory not found: {self.skills_dir}")

        for skill_folder in sorted(self.skills_dir.iterdir()):
            if not skill_folder.is_dir() or skill_folder.name.startswith("."):
                continue

            skill_dep = self._analyze_skill(skill_folder)
            if skill_dep:
                self.skills[skill_dep.name] = skill_dep

        return self.skills

    def _analyze_skill(self, skill_folder: Path) -> SkillDependency | None:
        """Analyze a single skill folder"""
        skill_name = skill_folder.name
        skill_dep = SkillDependency(
            name=skill_name,
            skill_path=str(skill_folder),
        )

        # 1. Parse requirements.txt
        requirements_file = skill_folder / "requirements.txt"
        if requirements_file.exists():
            skill_dep.pip_packages = self._parse_requirements(requirements_file)

        # 2. Scan SKILL.md for system dependencies
        skill_md = skill_folder / skill_name / "SKILL.md"
        if skill_md.exists():
            system_deps = self._extract_system_deps(skill_md)
            skill_dep.system_deps = system_deps

        # 3. Identify conda-only packages
        skill_dep.conda_packages = self._identify_conda_packages(skill_dep.pip_packages)

        return skill_dep

    def _parse_requirements(self, requirements_file: Path) -> list[str]:
        """Parse requirements.txt file"""
        packages = []
        try:
            content = requirements_file.read_text(encoding="utf-8")
            for line in content.splitlines():
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue
                # Extract package name (remove version specifiers)
                match = re.match(r"^([a-zA-Z0-9_-]+)(\[.*\])?", line)
                if match:
                    packages.append(match.group(0))
        except Exception:
            pass
        return packages

    def _extract_system_deps(self, skill_md: Path) -> list[str]:
        """Extract system dependencies from SKILL.md"""
        system_deps = set()
        try:
            content = skill_md.read_text(encoding="utf-8").lower()
            for pattern, conda_pkg in SYSTEM_DEP_PATTERNS.items():
                if re.search(pattern, content):
                    system_deps.add(conda_pkg)
        except Exception:
            pass
        return list(system_deps)

    def _identify_conda_packages(self, pip_packages: list[str]) -> list[str]:
        """Identify packages that should be installed via conda"""
        # Packages that are better installed via conda
        conda_preferred = {
            "rdkit": "rdkit",
            "pysam": "pysam",
            "pymatgen": "pymatgen",
            "biopython": "biopython",
            "anndata": "anndata",
            "scanpy": "scanpy",
            "cobra": "cobra",
            "scikit-bio": "scikit-bio",
        }

        conda_packages = []
        for pkg in pip_packages:
            pkg_lower = pkg.lower().split("[")[0].split(">=")[0].split("==")[0]
            if pkg_lower in conda_preferred:
                conda_packages.append(conda_preferred[pkg_lower])

        return conda_packages

    def to_json(self, output_file: str | None = None) -> str:
        """
        Export analysis results to JSON.

        Args:
            output_file: Optional output file path

        Returns:
            JSON string
        """
        data = {
            "skills_dir": str(self.skills_dir),
            "total_skills": len(self.skills),
            "skills": {name: dep.to_dict() for name, dep in self.skills.items()},
        }

        json_str = json.dumps(data, indent=2, ensure_ascii=False)

        if output_file:
            Path(output_file).write_text(json_str, encoding="utf-8")

        return json_str

    def print_summary(self) -> None:
        """Print a summary of analyzed skills"""
        print(f"\n{'='*60}")
        print(f"Skill Dependency Analysis Summary")
        print(f"{'='*60}")
        print(f"Skills Directory: {self.skills_dir}")
        print(f"Total Skills: {len(self.skills)}")
        print()

        # Collect all unique packages
        all_pip = set()
        all_conda = set()
        all_system = set()

        for dep in self.skills.values():
            all_pip.update(dep.pip_packages)
            all_conda.update(dep.conda_packages)
            all_system.update(dep.system_deps)

        print(f"Unique pip packages: {len(all_pip)}")
        print(f"Unique conda packages: {len(all_conda)}")
        print(f"Unique system deps: {len(all_system)}")

        if all_system:
            print(f"\nSystem dependencies found: {', '.join(sorted(all_system))}")

        print()
        print("Skills by dependency count:")
        for name, dep in sorted(self.skills.items(), key=lambda x: len(x[1].pip_packages), reverse=True)[:10]:
            print(f"  {name}: {len(dep.pip_packages)} pip packages")


def main():
    """CLI entry point for standalone testing"""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze skill dependencies")
    parser.add_argument(
        "--skills-dir",
        type=str,
        default="./skills",
        help="Path to skills directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output JSON file path",
    )

    args = parser.parse_args()

    analyzer = SkillDependencyAnalyzer(args.skills_dir)
    analyzer.analyze_all()
    analyzer.print_summary()

    if args.output:
        analyzer.to_json(args.output)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
