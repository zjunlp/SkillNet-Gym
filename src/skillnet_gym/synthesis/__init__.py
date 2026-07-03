"""SkillNet-Gym Task Auto-Synthesis Pipeline (DAG-aware).

Given input files, skill definitions, and an optional DAG task specification,
this pipeline explores the skills autonomously via Claude Code executions
and produces a fully verifiable task package (instruction, solve.sh, tests,
Dockerfile).
"""

from .config import (
    DAGEdge,
    DAGExplorationState,
    DAGTask,
    ExecutionConfig,
    ExplorationState,
    FileOperation,
    FileOperationType,
    HarborTask,
    PipelineConfig,
    ProcessedTrajectory,
    PromptConfig,
    TOOL_CATEGORY_MAP,
    ToolCategory,
    Trajectory,
    TrajectoryStep,
    ValidationResult,
)
from .pipeline import HarborSynthesisPipeline, HarborSynthesisPipelineV2

__all__ = [
    "PipelineConfig",
    "PromptConfig",
    "ExecutionConfig",
    "Trajectory",
    "TrajectoryStep",
    "FileOperation",
    "FileOperationType",
    "ToolCategory",
    "TOOL_CATEGORY_MAP",
    "ProcessedTrajectory",
    "ValidationResult",
    "HarborTask",
    "ExplorationState",
    "DAGTask",
    "DAGEdge",
    "DAGExplorationState",
    "HarborSynthesisPipeline",
    "HarborSynthesisPipelineV2",
]
