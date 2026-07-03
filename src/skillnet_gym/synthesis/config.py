"""Configuration and data structures for Harbor synthesis pipeline V2 (DAG-aware)"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from itertools import product
from typing import Any


class PromptStyle(str, Enum):
    """Prompt template styles for guiding Claude Code execution"""
    MINIMAL = "minimal"
    DOMAIN_GUIDED = "domain_guided"
    SKILL_HINTED = "skill_hinted"
    GOAL_ORIENTED = "goal_oriented"


class CreativityLevel(str, Enum):
    """Creativity level affecting prompt openness"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FileOperationType(str, Enum):
    """Types of file operations in trajectory"""
    CREATE = "create"
    READ = "read"
    EDIT = "edit"
    DELETE = "delete"
    NOTEBOOK_EDIT = "notebook_edit"


class ToolCategory(str, Enum):
    """Categories of Claude Code tools"""
    FILE_READ = "file_read"          # Read, Glob, Grep
    FILE_WRITE = "file_write"        # Write, Edit, NotebookEdit
    COMMAND = "command"              # Bash
    SKILL = "skill"                  # Skill
    WEB = "web"                      # WebFetch, WebSearch
    TASK = "task"                    # Task, TaskOutput, TaskStop, TodoWrite
    PLAN = "plan"                    # EnterPlanMode, ExitPlanMode
    WORKTREE = "worktree"            # EnterWorktree, ExitWorktree
    CRON = "cron"                    # CronCreate, CronDelete, CronList
    INTERACTION = "interaction"      # AskUserQuestion
    OTHER = "other"


class ContentType(str, Enum):
    """Content types for file classification (based on content, not extension)"""
    FORM = "form"       # Files with fillable fields, input boxes, checkboxes
    TEXT = "text"       # Pure text documents, articles, reports
    TABLE = "table"     # Files primarily containing tabular data
    CODE = "code"       # Source code files, scripts
    MIXED = "mixed"     # Files combining multiple content types


# Tool to category mapping
TOOL_CATEGORY_MAP: dict[str, ToolCategory] = {
    # File read operations
    "Read": ToolCategory.FILE_READ,
    "Glob": ToolCategory.FILE_READ,
    "Grep": ToolCategory.FILE_READ,
    # File write operations
    "Write": ToolCategory.FILE_WRITE,
    "Edit": ToolCategory.FILE_WRITE,
    "NotebookEdit": ToolCategory.FILE_WRITE,
    # Command execution
    "Bash": ToolCategory.COMMAND,
    # Skill invocation
    "Skill": ToolCategory.SKILL,
    # Web operations
    "WebFetch": ToolCategory.WEB,
    "WebSearch": ToolCategory.WEB,
    # Task management
    "Task": ToolCategory.TASK,
    "TaskOutput": ToolCategory.TASK,
    "TaskStop": ToolCategory.TASK,
    "TodoWrite": ToolCategory.TASK,
    # Plan mode
    "EnterPlanMode": ToolCategory.PLAN,
    "ExitPlanMode": ToolCategory.PLAN,
    # Worktree
    "EnterWorktree": ToolCategory.WORKTREE,
    "ExitWorktree": ToolCategory.WORKTREE,
    # Cron jobs
    "CronCreate": ToolCategory.CRON,
    "CronDelete": ToolCategory.CRON,
    "CronList": ToolCategory.CRON,
    # User interaction
    "AskUserQuestion": ToolCategory.INTERACTION,
}


@dataclass
class PromptConfig:
    """Configuration for prompt building"""
    template_style: PromptStyle = PromptStyle.MINIMAL
    max_steps: int = 10
    include_skills_hint: bool = False
    include_domain: bool = False
    creativity_level: CreativityLevel = CreativityLevel.MEDIUM
    custom_template: str | None = None


@dataclass
class ExecutionConfig:
    """Configuration for a single execution"""
    input_file: str | None = None
    skills_dir: str | None = None
    working_dir: str | None = None
    output_dir: str = "/root/output"
    prompt_config: PromptConfig = field(default_factory=PromptConfig)
    domain: str | None = None
    skills_hint: str | None = None


@dataclass
class PipelineConfig:
    """Main pipeline configuration"""
    # Model settings
    model: str = "claude-opus-4-5-20251124"
    max_turns: int = 50   # Increased for adaptive exploration
    timeout: int = 900    # 15 minutes for deep exploration

    # LLM client settings (shared API key/base_url). Values default to the
    # standard environment variables so users don't have to hard-code anything.
    llm_api_key: str = field(
        default_factory=lambda: __import__("os").environ.get("LLM_API_KEY")
        or __import__("os").environ.get("OPENAI_API_KEY", "")
    )
    llm_base_url: str = field(
        default_factory=lambda: __import__("os").environ.get(
            "LLM_BASE_URL", "https://api.openai.com/v1"
        )
    )
    llm_model: str = "gpt-4o"

    # Per-role model overrides (all use same api_key/base_url)
    # Instruction synthesis: generate_task_from_dag, filter, guiding_metadata
    llm_model_synthesis: str = "gpt-4o"
    # Verification: PRM validation, pytest generation, computation tests
    llm_model_verification: str = "gpt-4o"

    # Conda environment for Claude Code execution
    conda_env: str = "base"  # Conda environment name (e.g., "base", "myenv")

    # Output settings
    output_dir: str = "./workspaces"

    # Validation settings
    require_output_files: bool = True
    min_steps: int = 1

    # Task metadata defaults
    default_author: str = "Auto-Synthesis Pipeline"
    default_difficulty: str = "medium"
    default_category: str = "general"

    # Phase 1 自适应探索配置
    max_exploration_chunks: int = 30        # 安全阀：最大 chunk 数（扩大以给 Claude 更多探索时间）
    min_coverage_threshold: float = 0.9     # 覆盖率终止阈值

    # 并发配置
    max_workers: int = 3                    # 默认并发数
    show_progress: bool = True              # 是否显示进度条
    convergence_chunks: int = 3             # 收敛检测：连续无进展的 chunk 数
    checkpoint_interval: int = 20           # 检查点间隔：每 N 步写一次 checkpoint
    enable_goal_driven: bool = True         # 启用 Goal 驱动执行

    # Phase 3 测试驱动验证配置
    enable_test_verification: bool = True   # 是否启用测试驱动验证
    max_verification_retries: int = 2       # 验证失败最大重试次数
    test_timeout: int = 120                 # 单次测试执行超时（秒）

    # Phase 3.5.5 solve.sh 验证配置
    enable_solve_verification: bool = True  # 是否启用 solve.sh 验证
    max_solve_retries: int = 3              # solve.sh 验证失败最大重试次数
    solve_timeout: int = 120                # solve.sh 执行超时（秒）

    # 轨迹记录配置
    max_tool_output_chars: int = 2000       # 单步输出最大字符数
    skip_first_user_query: bool = True      # 跳过第一步 user query

    # solve.sh 生成配置
    use_claude_for_solve: bool = True       # 使用 Claude Code 生成 solve.sh（否则使用 LLM）

    # Phase 3.4.5 PRM 轨迹验证配置
    enable_prm_validation: bool = True      # 是否启用 PRM 轨迹验证
    max_prm_retries: int = 3                # PRM 验证失败最大重试次数
    prm_max_response_chars: int = 15000     # PRM 中 tool_response 超过此长度则截断

    # Phase 3.4.7 pytest 生成配置
    max_pytest_retries: int = 3             # pytest 生成失败最大重试次数

    # Phase 3.4.6-3.4.7 计算验证配置
    enable_computation_tests: bool = True   # 是否启用计算代码验证
    max_computation_retries: int = 5        # 计算测试失败最大重试次数

    # 路径规范化配置
    enable_path_normalization: bool = True  # 是否启用 solve.sh 路径规范化

    # DAG exploration config (V2)
    max_dag_exploration_chunks: int = 20        # Max exploration chunks for DAG
    dag_path_coverage_threshold: float = 1.0    # 100% path coverage required
    dag_convergence_chunks: int = 3             # Convergence detection for DAG


@dataclass
class TrajectoryStep:
    """A single step in the execution trajectory"""
    step_id: int
    action_type: str  # "tool_use", "text"
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: str | None = None
    timestamp: str = ""
    reasoning: str | None = None

    def is_file_operation(self) -> bool:
        """Check if this step is a file operation"""
        return self.tool_name in ("Write", "Edit", "Read", "NotebookEdit")

    def is_write_operation(self) -> bool:
        """Check if this step creates or modifies files"""
        return self.tool_name in ("Write", "Edit", "NotebookEdit")

    def is_read_operation(self) -> bool:
        """Check if this step reads files or searches"""
        return self.tool_name in ("Read", "Glob", "Grep")

    def is_skill_operation(self) -> bool:
        """Check if this step invokes a skill"""
        return self.tool_name == "Skill"

    def is_command_operation(self) -> bool:
        """Check if this step executes a command"""
        return self.tool_name == "Bash"

    def is_web_operation(self) -> bool:
        """Check if this step involves web access"""
        return self.tool_name in ("WebFetch", "WebSearch")

    @property
    def tool_category(self) -> "ToolCategory":
        """Get the category of this tool"""
        if self.tool_name:
            return TOOL_CATEGORY_MAP.get(self.tool_name, ToolCategory.OTHER)
        return ToolCategory.OTHER


@dataclass
class FileOperation:
    """Represents a file operation extracted from trajectory"""
    operation_type: FileOperationType
    file_path: str
    content: str | None = None
    timestamp: str = ""

    @property
    def filename(self) -> str:
        """Get the filename from the path"""
        return self.file_path.split("/")[-1]

    @property
    def extension(self) -> str:
        """Get the file extension"""
        if "." in self.filename:
            return self.filename.rsplit(".", 1)[-1].lower()
        return ""


@dataclass
class Trajectory:
    """Complete execution trajectory"""
    session_id: str
    steps: list[TrajectoryStep]
    input_files: list[str]
    output_files: list[str]
    success: bool
    duration_ms: int
    model: str
    raw_events: list[dict[str, Any]] = field(default_factory=list)
    # Extended tracking fields
    used_skills: list[str] = field(default_factory=list)
    bash_commands: list[str] = field(default_factory=list)
    web_urls: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    glob_patterns: list[str] = field(default_factory=list)
    grep_patterns: list[str] = field(default_factory=list)
    edited_files: list[str] = field(default_factory=list)
    notebook_files: list[str] = field(default_factory=list)

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def tool_use_steps(self) -> list[TrajectoryStep]:
        """Get only tool use steps"""
        return [s for s in self.steps if s.action_type == "tool_use"]

    @property
    def write_operations(self) -> list[TrajectoryStep]:
        """Get only write/edit operations"""
        return [s for s in self.steps if s.is_write_operation()]

    @property
    def all_modified_files(self) -> list[str]:
        """Get all files that were created or modified"""
        return list(dict.fromkeys(self.output_files + self.edited_files + self.notebook_files))


@dataclass
class ValidationResult:
    """Result of trajectory validation"""
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, error: str):
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str):
        self.warnings.append(warning)


@dataclass
class ProcessedTrajectory:
    """Trajectory with processed and extracted information"""
    trajectory: Trajectory
    file_operations: list[FileOperation]
    used_skills: list[str]
    pip_packages: list[str]
    summary: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def created_files(self) -> list[str]:
        """Get paths of created files"""
        return [
            op.file_path for op in self.file_operations
            if op.operation_type == FileOperationType.CREATE
        ]

    @property
    def output_file_types(self) -> set[str]:
        """Get set of output file extensions"""
        return {op.extension for op in self.file_operations if op.extension}


@dataclass
class HarborTask:
    """Generated Harbor task"""
    task_id: str
    task_path: str
    instruction: str
    input_files: list[str]
    output_files: list[str]
    used_skills: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        if not self.metadata:
            self.metadata = {}


@dataclass
class FileSummaryEntry:
    """A single file's summary with content type classification"""
    name: str              # File name
    path: str              # Full path
    summary: str           # File content summary
    content_type: str      # Content type (form, text, table, code, mixed)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary"""
        return {
            "name": self.name,
            "path": self.path,
            "summary": self.summary,
            "content_type": self.content_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileSummaryEntry":
        """Create from dictionary"""
        return cls(
            name=data["name"],
            path=data["path"],
            summary=data["summary"],
            content_type=data["content_type"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class FileSummaryResult:
    """Result of file summarization for a folder"""
    files: list[FileSummaryEntry] = field(default_factory=list)
    content_types: dict[str, list[str]] = field(default_factory=dict)  # content_type -> file paths
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary"""
        return {
            "files": [f.to_dict() for f in self.files],
            "content_types": self.content_types,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileSummaryResult":
        """Create from dictionary"""
        files = [FileSummaryEntry.from_dict(f) for f in data.get("files", [])]
        return cls(
            files=files,
            content_types=data.get("content_types", {}),
            created_at=data.get("created_at", ""),
        )

    def get_files_by_type(self, content_type: str) -> list[FileSummaryEntry]:
        """Get all files of a specific content type"""
        return [f for f in self.files if f.content_type == content_type]

    def filter_by_paths(self, paths: list[str]) -> "FileSummaryResult":
        """Create a new result with only the specified file paths"""
        path_set = set(paths)
        filtered_files = [f for f in self.files if f.path in path_set]

        # Rebuild content_types
        new_content_types: dict[str, list[str]] = {}
        for f in filtered_files:
            if f.content_type not in new_content_types:
                new_content_types[f.content_type] = []
            new_content_types[f.content_type].append(f.path)

        return FileSummaryResult(
            files=filtered_files,
            content_types=new_content_types,
            created_at=self.created_at,
        )


@dataclass
class LoopSummary:
    """单次循环探索的总结（程序化提取）"""
    loop_index: int
    trajectory: "Trajectory"
    # 程序化提取的字段
    used_skills: list[str] = field(default_factory=list)
    bash_commands: list[str] = field(default_factory=list)
    read_files: list[str] = field(default_factory=list)
    written_files: list[str] = field(default_factory=list)
    edited_files: list[str] = field(default_factory=list)
    tool_usage: dict[str, int] = field(default_factory=dict)  # tool_name -> count


@dataclass
class ExplorationSummaryTable:
    """循环探索的总结表"""
    loop_summaries: list[LoopSummary] = field(default_factory=list)
    total_skills_used: list[str] = field(default_factory=list)      # 去重后的所有技能
    all_bash_commands: list[str] = field(default_factory=list)       # 所有执行的命令
    all_read_files: list[str] = field(default_factory=list)          # 所有读取的文件
    all_written_files: list[str] = field(default_factory=list)       # 所有写入的文件
    total_tool_usage: dict[str, int] = field(default_factory=dict)   # 累计工具使用
    generated_goal: str | None = None


@dataclass
class ExplorationState:
    """可持久化的探索状态，用于自适应探索的检查点机制，支持多 skills"""
    # 基础信息
    skill_names: list[str] = field(default_factory=list)  # 支持多个 skills
    session_id: str = ""
    started_at: str = ""

    # 探索进度
    total_steps: int = 0
    checkpoint_count: int = 0

    # 覆盖率追踪（按 skill 分组）
    # 格式: {"skill_name": ["func1", "func2", ...]}
    documented_functions: dict[str, list[str]] = field(default_factory=dict)  # 从 SKILL.md 解析
    discovered_functions: dict[str, list[str]] = field(default_factory=dict)  # 探索中发现的
    tested_functions: dict[str, list[str]] = field(default_factory=dict)      # 已测试的

    # 工作流发现
    workflows: list[dict[str, Any]] = field(default_factory=list)  # [{name, steps, skills, tested}]

    # 收敛检测
    last_new_discovery_step: int = 0   # 上次发现新东西的步数
    consecutive_no_progress: int = 0   # 连续无进展的 chunk 数

    # 终止信号
    exploration_complete: bool = False
    completion_reason: str = ""

    @property
    def coverage_ratio(self) -> float:
        """计算所有 skills 的总覆盖率"""
        total_documented = sum(len(funcs) for funcs in self.documented_functions.values())
        if total_documented == 0:
            return 1.0

        total_tested = 0
        for skill, doc_funcs in self.documented_functions.items():
            tested_funcs = set(self.tested_functions.get(skill, []))
            total_tested += len(tested_funcs & set(doc_funcs))

        return total_tested / total_documented

    @property
    def coverage_by_skill(self) -> dict[str, float]:
        """计算每个 skill 的覆盖率"""
        coverage = {}
        for skill, doc_funcs in self.documented_functions.items():
            if not doc_funcs:
                coverage[skill] = 1.0
                continue
            tested_funcs = set(self.tested_functions.get(skill, []))
            coverage[skill] = len(tested_funcs & set(doc_funcs)) / len(doc_funcs)
        return coverage

    @property
    def should_continue(self) -> bool:
        """判断是否应该继续探索"""
        if self.exploration_complete:
            return False
        return True

    @property
    def total_documented_count(self) -> int:
        """文档化函数总数"""
        return sum(len(funcs) for funcs in self.documented_functions.values())

    @property
    def total_tested_count(self) -> int:
        """已测试函数总数"""
        total = 0
        for skill, doc_funcs in self.documented_functions.items():
            tested_funcs = set(self.tested_functions.get(skill, []))
            total += len(tested_funcs & set(doc_funcs))
        return total

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的字典"""
        return {
            "skill_names": self.skill_names,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "total_steps": self.total_steps,
            "checkpoint_count": self.checkpoint_count,
            "documented_functions": self.documented_functions,
            "discovered_functions": self.discovered_functions,
            "tested_functions": self.tested_functions,
            "workflows": self.workflows,
            "last_new_discovery_step": self.last_new_discovery_step,
            "consecutive_no_progress": self.consecutive_no_progress,
            "exploration_complete": self.exploration_complete,
            "completion_reason": self.completion_reason,
            "coverage_ratio": self.coverage_ratio,
            "coverage_by_skill": self.coverage_by_skill,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExplorationState":
        """从字典创建 ExplorationState"""
        from dataclasses import fields as dataclass_fields

        # 获取 dataclass 定义的字段名
        valid_field_names = {f.name for f in dataclass_fields(cls)}

        # 移除计算属性和未知字段（Claude 可能自己添加的字段）
        computed_props = ("coverage_ratio", "coverage_by_skill", "should_continue",
                         "total_documented_count", "total_tested_count")
        data = {k: v for k, v in data.items()
                if k not in computed_props and k in valid_field_names}

        # 向后兼容：处理旧格式 skill_name -> skill_names
        if "skill_name" in data and "skill_names" not in data:
            skill_name = data.pop("skill_name")
            data["skill_names"] = [skill_name] if skill_name else []

        # 向后兼容：处理旧格式 list -> dict
        for field_name in ("documented_functions", "discovered_functions", "tested_functions"):
            if field_name in data and isinstance(data[field_name], list):
                # 旧格式是 list，转为 dict（放在 "unknown" skill 下）
                old_list = data[field_name]
                skill_names = data.get("skill_names", ["unknown"])
                if skill_names:
                    data[field_name] = {skill_names[0]: old_list}
                else:
                    data[field_name] = {"unknown": old_list}

        return cls(**data)


class InvalidTrajectoryError(Exception):
    """Raised when trajectory validation fails"""
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"Invalid trajectory: {', '.join(errors)}")


class SynthesisError(Exception):
    """Raised when task synthesis fails"""
    pass


class SolveSynthesisError(SynthesisError):
    """Raised when solve.sh generation/validation fails after max retries"""
    pass


# =========================================================================
# DAG-specific data structures (V2)
# =========================================================================

@dataclass
class DAGEdge:
    """A directed edge in the skill DAG."""
    source_skill: str
    target_skill: str
    alignment_types: list[str] = field(default_factory=list)
    scenario_description: str = ""
    edge_score: float = 0.0

    def __str__(self) -> str:
        return f"{self.source_skill} -> {self.target_skill}"


@dataclass
class DAGTask:
    """A task defined by a DAG of skills."""
    task_id: str
    structure_type: str  # chain, fan_out, fan_in, diamond
    category: str
    difficulty: str
    skills: list[dict[str, Any]]  # [{skill_id, skill_name}]
    edges: list[DAGEdge]
    skill_connections: list[str]  # ["A -> B", ...]
    suggested_task: dict[str, Any]  # {title, input, workflow, expected_output, verification}
    skill_doc_paths: dict[str, str] = field(default_factory=dict)  # skill_name -> SKILL.md path

    @property
    def skill_names(self) -> list[str]:
        return [s["skill_name"] for s in self.skills]

    @property
    def source_nodes(self) -> list[str]:
        """Nodes with no incoming edges."""
        targets = {e.target_skill for e in self.edges}
        return [n for n in self.skill_names if n not in targets]

    @property
    def sink_nodes(self) -> list[str]:
        """Nodes with no outgoing edges."""
        sources = {e.source_skill for e in self.edges}
        return [n for n in self.skill_names if n not in sources]

    def topological_order(self) -> list[str]:
        """Kahn's algorithm for topological sort."""
        adj: dict[str, list[str]] = {n: [] for n in self.skill_names}
        in_deg: dict[str, int] = {n: 0 for n in self.skill_names}
        for e in self.edges:
            adj[e.source_skill].append(e.target_skill)
            in_deg[e.target_skill] += 1
        queue = [n for n in self.skill_names if in_deg[n] == 0]
        order: list[str] = []
        while queue:
            queue.sort()
            node = queue.pop(0)
            order.append(node)
            for nb in adj[node]:
                in_deg[nb] -= 1
                if in_deg[nb] == 0:
                    queue.append(nb)
        return order

    def all_paths(self) -> list[list[str]]:
        """Enumerate all source-to-sink paths via DFS."""
        adj: dict[str, list[str]] = {n: [] for n in self.skill_names}
        for e in self.edges:
            adj[e.source_skill].append(e.target_skill)
        sinks = set(self.sink_nodes)
        result: list[list[str]] = []

        def dfs(node: str, path: list[str]) -> None:
            if node in sinks and len(path) > 1:
                result.append(list(path))
                if adj[node]:
                    for nb in adj[node]:
                        path.append(nb)
                        dfs(nb, path)
                        path.pop()
                return
            for nb in adj[node]:
                path.append(nb)
                dfs(nb, path)
                path.pop()

        for src in self.source_nodes:
            dfs(src, [src])

        if not result:
            for n in self.skill_names:
                result.append([n])
        return result

    def get_edge(self, source: str, target: str) -> DAGEdge | None:
        for e in self.edges:
            if e.source_skill == source and e.target_skill == target:
                return e
        return None

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "DAGTask":
        """Parse a DAG task from the skill_graph_task JSON format."""
        edges = []
        for edge_data in data.get("edges", []):
            sc = edge_data.get("scenario_connections", [])
            desc = sc[0].get("reason", "") if sc else ""
            edges.append(DAGEdge(
                source_skill=edge_data["source_skill_name"],
                target_skill=edge_data["target_skill_name"],
                alignment_types=edge_data.get("alignment_types", []),
                scenario_description=desc,
                edge_score=edge_data.get("edge_score", 0.0),
            ))

        return cls(
            task_id=data.get("task_id", f"dag_{data.get('category', 'unknown')}"),
            structure_type=data.get("structure_type", "chain"),
            category=data.get("category", ""),
            difficulty=data.get("difficulty", "medium"),
            skills=data.get("skills", []),
            edges=edges,
            skill_connections=data.get("skill_connections", []),
            suggested_task=data.get("suggested_task", {}),
            skill_doc_paths=data.get("skill_doc_paths", {}),
        )


@dataclass
class DAGExplorationState:
    """Exploration state tracking for DAG-based exploration."""
    dag_task_id: str = ""
    skill_names: list[str] = field(default_factory=list)

    # Path coverage (100% required)
    all_paths: list[list[str]] = field(default_factory=list)
    tested_paths: list[list[str]] = field(default_factory=list)

    # Edge coverage
    all_edges: list[str] = field(default_factory=list)  # ["A->B", ...]
    tested_edges: list[str] = field(default_factory=list)

    # Per-skill exploration
    core_functions: dict[str, list[str]] = field(default_factory=dict)
    tested_core_functions: dict[str, list[str]] = field(default_factory=dict)
    other_functions: dict[str, list[str]] = field(default_factory=dict)
    tested_other_functions: dict[str, list[str]] = field(default_factory=dict)

    # State
    total_steps: int = 0
    checkpoint_count: int = 0
    exploration_complete: bool = False
    completion_reason: str = ""

    @property
    def path_coverage(self) -> float:
        if not self.all_paths:
            return 1.0
        tested_set = {tuple(p) for p in self.tested_paths}
        return len(tested_set) / len(self.all_paths)

    @property
    def edge_coverage(self) -> float:
        if not self.all_edges:
            return 1.0
        return len(set(self.tested_edges)) / len(set(self.all_edges))

    @property
    def core_function_coverage(self) -> float:
        total = sum(len(v) for v in self.core_functions.values())
        if total == 0:
            return 1.0
        tested = 0
        for skill, funcs in self.core_functions.items():
            tested += len(set(self.tested_core_functions.get(skill, [])) & set(funcs))
        return tested / total

    @property
    def should_continue(self) -> bool:
        if self.exploration_complete:
            return False
        return self.path_coverage < 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dag_task_id": self.dag_task_id,
            "skill_names": self.skill_names,
            "all_paths": self.all_paths,
            "tested_paths": self.tested_paths,
            "all_edges": self.all_edges,
            "tested_edges": self.tested_edges,
            "core_functions": self.core_functions,
            "tested_core_functions": self.tested_core_functions,
            "other_functions": self.other_functions,
            "tested_other_functions": self.tested_other_functions,
            "total_steps": self.total_steps,
            "checkpoint_count": self.checkpoint_count,
            "exploration_complete": self.exploration_complete,
            "completion_reason": self.completion_reason,
            "path_coverage": self.path_coverage,
            "edge_coverage": self.edge_coverage,
            "core_function_coverage": self.core_function_coverage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DAGExplorationState":
        from dataclasses import fields as dataclass_fields
        valid_field_names = {f.name for f in dataclass_fields(cls)}
        computed = ("path_coverage", "edge_coverage", "core_function_coverage", "should_continue")
        clean = {k: v for k, v in data.items() if k not in computed and k in valid_field_names}
        return cls(**clean)

    @classmethod
    def from_dag_task(cls, dag_task: "DAGTask") -> "DAGExplorationState":
        """Initialize from a DAGTask, computing all_paths and all_edges."""
        paths = dag_task.all_paths()
        edges = [f"{e.source_skill}->{e.target_skill}" for e in dag_task.edges]
        return cls(
            dag_task_id=dag_task.task_id,
            skill_names=dag_task.skill_names,
            all_paths=paths,
            all_edges=edges,
        )
