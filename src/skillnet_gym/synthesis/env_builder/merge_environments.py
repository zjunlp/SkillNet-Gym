"""
Environment Merger

Merges compatible skills into shared conda environments.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .analyze_skill_deps import SkillDependency, SkillDependencyAnalyzer


# Environment merge rules - defines which skills belong to which environment
# Format: env_name -> list of skill name patterns (supports prefix matching)
ENV_MERGE_RULES: dict[str, dict[str, Any]] = {
    "office": {
        "description": "Document processing (PDF, Excel, Word, PowerPoint)",
        "skills": [
            "pdf",
            "xlsx",
            "docx",
            "pptx",
            "markdown-to-html",
            "invoice-organizer",
            "Finance Skill",
            "marker",
            "openakitaskills@translate-pdf",
        ],
        "conda_deps": ["poppler", "tesseract"],
        "python_version": "3.10",
    },
    "data": {
        "description": "Data analysis and machine learning",
        "skills": [
            "Data Analysis",
            "Excel Analysis",
            "Pandas Construction Analysis",
            "data-transform",
            "scikit-learn",
            "statsmodels",
            "statistical-analysis",
            "polars",
            "networkx",
            "contribution-analysis",
            "pca-decomposition",
            "ab-unit-harmonization",
            "lab-unit-harmonization",
            "did_causal_analysis",
            "dividend-tracking",
            "search-flights",
            "xsv",
        ],
        "conda_deps": [],
        "python_version": "3.10",
    },
    "bio": {
        "description": "Bioinformatics and computational biology",
        "skills": [
            "biopython",
            "Bioinformatics",
            "pysam",
            "pydeseq2",
            "scikit-bio",
            "anndata",
            "cobrapy",
        ],
        "conda_deps": ["pysam", "biopython"],
        "python_version": "3.10",
    },
    "chem": {
        "description": "Chemistry and molecular modeling",
        "skills": [
            "rdkit",
            "datamol",
            "pymatgen",
        ],
        "conda_deps": ["rdkit"],
        "python_version": "3.10",
    },
    "geo": {
        "description": "Geospatial analysis",
        "skills": [
            "geopandas",
            "geospatial-analysis",
            "flood-detection",
        ],
        "conda_deps": ["geopandas", "gdal"],
        "python_version": "3.10",
    },
    "astro": {
        "description": "Astronomy and time series",
        "skills": [
            "astropy",
            "aeon",
        ],
        "conda_deps": [],
        "python_version": "3.10",
    },
    "neuro": {
        "description": "Neuroscience and physiological signals",
        "skills": [
            "neurokit2",
        ],
        "conda_deps": [],
        "python_version": "3.10",
    },
    "medical": {
        "description": "Medical imaging",
        "skills": [
            "pydicom",
        ],
        "conda_deps": [],
        "python_version": "3.10",
    },
    "network": {
        "description": "Network analysis and security",
        "skills": [
            "pcap-analysis",
            "HTML Injection Testing",
        ],
        "conda_deps": [],
        "python_version": "3.10",
    },
    "audio": {
        "description": "Audio processing",
        "skills": [
            "audiobook",
        ],
        "conda_deps": ["ffmpeg"],
        "python_version": "3.10",
    },
    "math": {
        "description": "Mathematical computing",
        "skills": [
            "math",
            "sympy",
            "dc-power-flow",
        ],
        "conda_deps": [],
        "python_version": "3.10",
    },
    "misc": {
        "description": "Miscellaneous skills",
        "skills": [
            "code-review",
        ],
        "conda_deps": [],
        "python_version": "3.10",
    },
}


@dataclass
class EnvironmentConfig:
    """Configuration for a conda environment"""
    name: str
    description: str
    python_version: str
    skills: list[str] = field(default_factory=list)
    pip_packages: list[str] = field(default_factory=list)
    conda_packages: list[str] = field(default_factory=list)
    system_deps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "python_version": self.python_version,
            "skills": self.skills,
            "pip_packages": sorted(set(self.pip_packages)),
            "conda_packages": sorted(set(self.conda_packages)),
            "system_deps": sorted(set(self.system_deps)),
        }


class EnvironmentMerger:
    """Merges skills into shared environments based on rules"""

    def __init__(
        self,
        skill_deps: dict[str, SkillDependency],
        merge_rules: dict[str, dict[str, Any]] | None = None,
    ):
        """
        Initialize merger.

        Args:
            skill_deps: Dictionary of skill dependencies from analyzer
            merge_rules: Custom merge rules (uses default if None)
        """
        self.skill_deps = skill_deps
        self.merge_rules = merge_rules or ENV_MERGE_RULES
        self.environments: dict[str, EnvironmentConfig] = {}
        self.skill_to_env: dict[str, str] = {}
        self.unassigned_skills: list[str] = []

    def merge(self) -> dict[str, EnvironmentConfig]:
        """
        Merge skills into environments.

        Returns:
            Dictionary of environment configurations
        """
        # Initialize environments from rules
        for env_name, rule in self.merge_rules.items():
            self.environments[env_name] = EnvironmentConfig(
                name=env_name,
                description=rule.get("description", ""),
                python_version=rule.get("python_version", "3.10"),
                conda_packages=rule.get("conda_deps", []).copy(),
            )

        # Assign skills to environments
        assigned_skills = set()
        for env_name, rule in self.merge_rules.items():
            for skill_pattern in rule.get("skills", []):
                # Find matching skills
                for skill_name, skill_dep in self.skill_deps.items():
                    if skill_name == skill_pattern or skill_name.startswith(skill_pattern):
                        if skill_name not in assigned_skills:
                            self._assign_skill_to_env(skill_name, skill_dep, env_name)
                            assigned_skills.add(skill_name)

        # Handle unassigned skills -> put in misc
        for skill_name, skill_dep in self.skill_deps.items():
            if skill_name not in assigned_skills:
                self.unassigned_skills.append(skill_name)
                self._assign_skill_to_env(skill_name, skill_dep, "misc")

        return self.environments

    def _assign_skill_to_env(
        self,
        skill_name: str,
        skill_dep: SkillDependency,
        env_name: str,
    ) -> None:
        """Assign a skill to an environment"""
        env = self.environments[env_name]
        env.skills.append(skill_name)
        env.pip_packages.extend(skill_dep.pip_packages)
        env.conda_packages.extend(skill_dep.conda_packages)
        env.system_deps.extend(skill_dep.system_deps)
        self.skill_to_env[skill_name] = env_name

    def generate_mapping(self, output_file: str | None = None) -> dict:
        """
        Generate skill-to-environment mapping.

        Args:
            output_file: Optional output file path

        Returns:
            Mapping dictionary
        """
        mapping = {
            "environments": {
                name: env.to_dict()
                for name, env in self.environments.items()
                if env.skills  # Only include environments with skills
            },
            "skill_to_env": self.skill_to_env,
            "unassigned_skills": self.unassigned_skills,
        }

        if output_file:
            json_str = json.dumps(mapping, indent=2, ensure_ascii=False)
            Path(output_file).write_text(json_str, encoding="utf-8")

        return mapping

    def print_summary(self) -> None:
        """Print merge summary"""
        print(f"\n{'='*60}")
        print("Environment Merge Summary")
        print(f"{'='*60}")

        for env_name, env in sorted(self.environments.items()):
            if not env.skills:
                continue
            print(f"\n[{env_name}] {env.description}")
            print(f"  Skills ({len(env.skills)}): {', '.join(sorted(env.skills))}")
            print(f"  Pip packages: {len(set(env.pip_packages))}")
            print(f"  Conda packages: {', '.join(sorted(set(env.conda_packages))) or 'none'}")
            if env.system_deps:
                print(f"  System deps: {', '.join(sorted(set(env.system_deps)))}")

        if self.unassigned_skills:
            print(f"\nUnassigned skills (added to misc): {', '.join(self.unassigned_skills)}")


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Merge skills into environments")
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
        help="Output mapping JSON file path",
    )

    args = parser.parse_args()

    # Analyze dependencies
    analyzer = SkillDependencyAnalyzer(args.skills_dir)
    analyzer.analyze_all()

    # Merge environments
    merger = EnvironmentMerger(analyzer.skills)
    merger.merge()
    merger.print_summary()

    if args.output:
        merger.generate_mapping(args.output)
        print(f"\nMapping saved to: {args.output}")


if __name__ == "__main__":
    main()
