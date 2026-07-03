"""Main pipeline for Harbor task synthesis"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import (
    DAGExplorationState,
    DAGTask,
    ExecutionConfig,
    ExplorationState,
    FileSummaryResult,
    HarborTask,
    InvalidTrajectoryError,
    PipelineConfig,
    PromptConfig,
    PromptStyle,
    SolveSynthesisError,
    SynthesisError,
    Trajectory,
)
from .execution.claude_executor import ClaudeExecutor
from .execution.prompt_builder import PromptBuilder
from .execution.trajectory_recorder import TrajectoryRecorder
from .components.exploration_summarizer import ExplorationSummarizer
from .components.file_summarizer import FileSummarizer
from .components.instruction_generator import InstructionGenerator
from .components.task_packager import TaskPackager
from .components.test_executor import TestExecutor
from .components.test_generator import TestGenerator
from .components.trajectory_processor import TrajectoryProcessor
from .components.solve_generator import SolveShGenerator
from .components.solve_verifier import SolveShVerifier
from .components.trajectory_validator import TrajectoryValidator, PRMValidationResult
from .components.computation_test_generator import ComputationTestGenerator
from .components.path_normalizer import PathNormalizer, create_path_normalizer
from .components.pytest_generator import PytestGenerator
from .utils.llm_client import LLMClient
from .utils.path_normalizer import normalize_for_skillsbench, SkillsbenchPathNormalizer


class HarborSynthesisPipeline:
    """Main pipeline for synthesizing Harbor tasks from Claude Code execution"""

    def __init__(self, config: PipelineConfig | None = None):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration, uses defaults if None
        """
        self.config = config or PipelineConfig()

        # Initialize components
        self.prompt_builder = PromptBuilder()
        self.executor = ClaudeExecutor.from_config(self.config)
        self.recorder = TrajectoryRecorder()
        self.processor = TrajectoryProcessor(
            require_output=self.config.require_output_files,
            min_steps=self.config.min_steps,
        )
        self.test_generator = TestGenerator()
        self.packager = TaskPackager(
            author_name=self.config.default_author,
            default_difficulty=self.config.default_difficulty,
            default_category=self.config.default_category,
        )

        # LLM client for instruction generation (lazy init)
        self._llm_client = None
        self._llm_client_synthesis = None
        self._llm_client_verification = None
        self._instruction_generator = None
        self._exploration_summarizer = None
        self._test_executor = None
        self._solve_generator = None
        self._solve_verifier = None
        self._trajectory_validator = None
        self._computation_test_generator = None
        self._pytest_generator = None

    @property
    def llm_client(self) -> LLMClient:
        """Lazy-initialize LLM client (default model)"""
        if self._llm_client is None:
            self._llm_client = LLMClient(
                api_key=self.config.llm_api_key,
                base_url=self.config.llm_base_url,
                model=self.config.llm_model,
            )
        return self._llm_client

    @property
    def llm_client_synthesis(self) -> LLMClient:
        """LLM client for instruction synthesis (gemini)"""
        if self._llm_client_synthesis is None:
            self._llm_client_synthesis = LLMClient(
                api_key=self.config.llm_api_key,
                base_url=self.config.llm_base_url,
                model=self.config.llm_model_synthesis,
            )
        return self._llm_client_synthesis

    @property
    def llm_client_verification(self) -> LLMClient:
        """LLM client for verification tasks (gpt-4o)"""
        if self._llm_client_verification is None:
            self._llm_client_verification = LLMClient(
                api_key=self.config.llm_api_key,
                base_url=self.config.llm_base_url,
                model=self.config.llm_model_verification,
            )
        return self._llm_client_verification

    @property
    def instruction_generator(self) -> InstructionGenerator:
        """Lazy-initialize instruction generator (uses synthesis model)"""
        if self._instruction_generator is None:
            self._instruction_generator = InstructionGenerator(self.llm_client_synthesis)
        return self._instruction_generator

    @property
    def exploration_summarizer(self) -> ExplorationSummarizer:
        """Lazy-initialize exploration summarizer (uses synthesis model)"""
        if self._exploration_summarizer is None:
            self._exploration_summarizer = ExplorationSummarizer(self.llm_client_synthesis)
        return self._exploration_summarizer

    @property
    def file_summarizer(self) -> FileSummarizer:
        """Lazy-initialize file summarizer (uses synthesis model)"""
        if not hasattr(self, "_file_summarizer") or self._file_summarizer is None:
            self._file_summarizer = FileSummarizer(
                executor=self.executor,
                llm_client=self.llm_client_synthesis,
                max_workers=self.config.max_workers,
                show_progress=self.config.show_progress,
            )
        return self._file_summarizer

    @property
    def test_executor(self) -> TestExecutor:
        """Lazy-initialize test executor"""
        if self._test_executor is None:
            self._test_executor = TestExecutor(conda_env=self.config.conda_env)
        return self._test_executor

    @property
    def solve_generator(self) -> SolveShGenerator:
        """Lazy-initialize solve.sh generator (uses synthesis model)"""
        if self._solve_generator is None:
            self._solve_generator = SolveShGenerator(llm_client=self.llm_client_synthesis)
        return self._solve_generator

    @property
    def solve_verifier(self) -> SolveShVerifier:
        """Lazy-initialize solve.sh verifier"""
        if self._solve_verifier is None:
            self._solve_verifier = SolveShVerifier(test_executor=self.test_executor)
        return self._solve_verifier

    @property
    def trajectory_validator(self) -> TrajectoryValidator:
        """Lazy-initialize trajectory validator (PRM, uses verification model)"""
        if self._trajectory_validator is None:
            self._trajectory_validator = TrajectoryValidator(
                executor=self.executor,
                llm_client=self.llm_client_verification,
                max_response_chars=self.config.prm_max_response_chars,
            )
        return self._trajectory_validator

    @property
    def computation_test_generator(self) -> ComputationTestGenerator:
        """Lazy-initialize computation test generator (uses verification model)"""
        if self._computation_test_generator is None:
            self._computation_test_generator = ComputationTestGenerator(
                executor=self.executor,
                llm_client=self.llm_client_verification,
            )
        return self._computation_test_generator

    @property
    def pytest_generator(self) -> PytestGenerator:
        """Lazy-initialize pytest generator (uses verification model)"""
        if self._pytest_generator is None:
            self._pytest_generator = PytestGenerator(
                executor=self.executor,
                llm_client=self.llm_client_verification,
            )
        return self._pytest_generator

    EXPLORATION_SUMMARY_FILE = "exploration_summary.md"
    EXPLORATION_STATE_FILE = "exploration_state.json"
    FILE_SUMMARY_FILE = "file_summaries.json"

    # =========================================================================
    # Phase 1: File Summary
    # =========================================================================

    def phase1_file_summary(
        self,
        entity_folder: str,
        output_json: str | None = None,
        file_extensions: list[str] | None = None,
        ignore_files: list[str] | None = None,
        extract_metadata: bool = False,
    ) -> FileSummaryResult:
        """
        阶段一：文件总结

        对实体文件夹下的全部文件进行总结，生成包含文件名、地址、摘要和内容类型的 JSON。

        Args:
            entity_folder: 实体文件夹路径
            output_json: 输出 JSON 文件路径（可选）
            file_extensions: 要处理的文件扩展名列表（如 ['.pdf', '.xlsx']）
            ignore_files: 要忽略的文件名列表（如 ['requirements.txt']）
            extract_metadata: 是否提取详细 metadata（默认 False）

        Returns:
            FileSummaryResult 对象
        """
        print(f"\n[Phase 1] File Summary Generation")
        print(f"[Phase 1] Entity folder: {entity_folder}")
        if ignore_files:
            print(f"[Phase 1] Ignoring files: {ignore_files}")
        if extract_metadata:
            print(f"[Phase 1] Metadata extraction: enabled")

        # Set extract_metadata flag on summarizer
        self.file_summarizer.extract_metadata = extract_metadata

        result = self.file_summarizer.summarize_folder(
            folder_path=entity_folder,
            output_json=output_json,
            file_extensions=file_extensions,
            ignore_files=ignore_files,
        )

        print(f"[Phase 1] Processed {len(result.files)} files")
        for ct, paths in result.content_types.items():
            print(f"[Phase 1]   {ct}: {len(paths)} files")

        return result

    # =========================================================================
    # Phase 2: Skill Exploration (with reuse support)
    # =========================================================================

    def phase2_exploration(
        self,
        file_summary: FileSummaryResult | str,
        skills_dir: str,
        output_dir: str,
        existing_exploration: str | None = None,
        use_all_files: bool = False,
    ) -> str:
        """
        阶段二：技能探索（支持复用）

        如果已有探索报告，直接复用；否则选择代表性文件执行新探索。

        Args:
            file_summary: FileSummaryResult 对象或 JSON 文件路径
            skills_dir: 技能目录路径
            output_dir: 输出目录路径
            existing_exploration: 已有的探索报告路径（用于复用）
            use_all_files: 是否使用所有文件（True 则跳过代表性文件选择）

        Returns:
            探索报告文件路径
        """
        print(f"\n[Phase 2] Skill Exploration")

        # 检查是否复用已有探索报告
        if existing_exploration and Path(existing_exploration).exists():
            print(f"[Phase 2] Reusing existing exploration: {existing_exploration}")
            return existing_exploration

        # 加载 file_summary
        if isinstance(file_summary, str):
            print(f"[Phase 2] Loading file summary from: {file_summary}")
            file_summary = self.file_summarizer.load_from_json(file_summary)

        # 选择文件：使用所有文件或选择代表性文件
        if use_all_files:
            # 使用所有文件
            representative_files = [entry.path for entry in file_summary.files]
            print(f"[Phase 2] Using all {len(representative_files)} files (--all-files mode)")
        else:
            # 选择代表性文件
            print(f"[Phase 2] Selecting representative files...")
            representative_files = self._select_representative_files_with_llm(file_summary)
            print(f"[Phase 2] Selected {len(representative_files)} representative files")

        if not representative_files:
            raise SynthesisError("No representative files selected for exploration")

        # 准备文件摘要（只包含选中的文件）
        rep_summaries = {
            entry.path: entry.summary
            for entry in file_summary.files
            if entry.path in representative_files
        }

        # 执行探索
        return self._execute_exploration(
            file_summaries=rep_summaries,
            skills_dir=skills_dir,
            output_dir=output_dir,
        )

    def _select_representative_files_with_llm(
        self,
        file_summary: FileSummaryResult,
    ) -> list[str]:
        """
        使用 LLM 根据文件总结选择代表性文件。

        每种内容类型选择一个最具代表性且内容丰富的文件。

        Args:
            file_summary: FileSummaryResult 对象

        Returns:
            选中的文件路径列表
        """
        from .prompts.representative_file_selection import (
            SYS_PROMPT,
            USER_PROMPT,
        )

        # 按内容类型组织文件摘要
        summaries_by_type: dict[str, list[dict]] = {}
        for entry in file_summary.files:
            ct = entry.content_type
            if ct not in summaries_by_type:
                summaries_by_type[ct] = []
            summaries_by_type[ct].append({
                "path": entry.path,
                "name": entry.name,
                "summary": entry.summary,
            })

        # 格式化为 prompt 输入
        formatted = ""
        for ct, files in summaries_by_type.items():
            formatted += f"\n## Content Type: {ct}\n"
            for f in files:
                formatted += f"\n### {f['name']}\n"
                formatted += f"Path: {f['path']}\n"
                formatted += f"Summary: {f['summary']}\n"

        # 调用 LLM
        prompt = USER_PROMPT.format(file_summaries_by_type=formatted)
        response = self.llm_client_synthesis.generate(
            system_prompt=SYS_PROMPT,
            user_prompt=prompt,
            temperature=0.3,
        )

        # 解析结果
        return self._parse_selection_response(response)

    def _parse_selection_response(self, response: str) -> list[str]:
        """
        解析 LLM 的文件选择响应。

        Args:
            response: LLM 响应

        Returns:
            选中的文件路径列表
        """
        import re

        selected_files = []

        # 尝试解析 JSON
        try:
            # 查找 JSON 块
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 尝试直接解析整个响应
                json_str = response

            data = json.loads(json_str)

            if "selections" in data:
                for sel in data["selections"]:
                    if "selected_file" in sel:
                        selected_files.append(sel["selected_file"])

        except json.JSONDecodeError:
            # Fallback: 从响应中提取文件路径
            path_pattern = r'["\']?(/[^"\'<>\s]+\.[a-zA-Z0-9]+)["\']?'
            matches = re.findall(path_pattern, response)
            selected_files = list(dict.fromkeys(matches))  # 去重保持顺序

        print(f"[Phase 2] Parsed {len(selected_files)} selected files")
        for f in selected_files:
            print(f"[Phase 2]   - {f}")

        return selected_files

    def _execute_exploration(
        self,
        file_summaries: dict[str, str],
        skills_dir: str,
        output_dir: str,
    ) -> str:
        """
        执行技能探索并返回探索报告路径。

        使用隔离工作空间确保 Claude Code 只能访问必需的文件，
        执行完成后将结果复制回目标目录并清理临时工作空间。

        Args:
            file_summaries: 文件路径到摘要的映射
            skills_dir: 技能目录路径
            output_dir: 输出目录路径

        Returns:
            探索报告文件路径
        """
        from .utils.file_utils import (
            setup_exploration_workspace,
            cleanup_exploration_workspace,
            copy_exploration_results,
        )

        # [FIX] 将 output_dir 转换为绝对路径
        output_dir = str(Path(output_dir).resolve())
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print(f"[Phase 2] Target output dir: {output_dir}")

        # =====================================================================
        # Step 1: 创建隔离工作空间
        # =====================================================================
        input_files = list(file_summaries.keys())
        workspace_info = setup_exploration_workspace(
            skills_dir=skills_dir,
            input_files=input_files,
            file_summaries=file_summaries,
        )
        workspace_path = workspace_info["workspace"]
        input_file_mapping = workspace_info.get("input_file_mapping", {})
        print(f"[Phase 2] Created isolated workspace: {workspace_path}")

        # 将 file_summaries 转换为使用工作空间路径
        workspace_file_summaries = {}
        for orig_path, summary in file_summaries.items():
            workspace_path_str = input_file_mapping.get(str(Path(orig_path).resolve()))
            if workspace_path_str:
                workspace_file_summaries[workspace_path_str] = summary
            else:
                # 文件未复制，使用原路径（fallback）
                workspace_file_summaries[orig_path] = summary

        # 工作目录设为工作空间的 output 子目录
        working_dir = str(workspace_info["output"]) if isinstance(workspace_info["output"], Path) else str(workspace_info["output"])

        # =====================================================================
        # Step 2: 初始化探索状态
        # =====================================================================
        skill_names = self._extract_skill_names(skills_dir)
        documented_functions = self._extract_documented_functions(skills_dir)

        initial_state = ExplorationState(
            skill_names=skill_names,
            session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            started_at=datetime.now().isoformat(),
            documented_functions=documented_functions,
        )
        # 状态保存到工作空间的 output 目录
        self._save_exploration_state(initial_state, working_dir)
        print(f"[Phase 2] Initial state saved to workspace: {working_dir}")

        # 使用第一个工作空间输入文件
        primary_input_file = next(iter(workspace_file_summaries.keys())) if workspace_file_summaries else None

        # =====================================================================
        # Step 3: 执行探索循环
        # =====================================================================
        max_chunks = self.config.max_exploration_chunks
        prompt_config = PromptConfig(
            template_style=PromptStyle.MINIMAL,
            max_steps=20,
        )
        self.prompt_builder = PromptBuilder(prompt_config)

        # 程序化收敛检测（不依赖 Claude 更新 consecutive_no_progress）
        programmatic_no_progress = 0
        last_tested_count = 0

        try:
            for chunk_idx in range(1, max_chunks + 1):
                print(f"\n[Phase 2.{chunk_idx}] Exploration chunk {chunk_idx}/{max_chunks}")

                current_state = self._read_exploration_state(working_dir) or initial_state

                # 记录执行前的 tested_functions 数量
                pre_tested_count = current_state.total_tested_count

                print(f"[Phase 2.{chunk_idx}] Current state: coverage={current_state.coverage_ratio:.1%}, "
                      f"complete={current_state.exploration_complete}, "
                      f"tested={pre_tested_count}, no_progress={programmatic_no_progress}")

                should_continue, reason = self._check_should_continue(current_state, chunk_idx - 1, working_dir)
                if not should_continue:
                    print(f"[Phase 2.{chunk_idx}] Stopping: {reason}")
                    break

                # 读取上一个 chunk 的 summary 备份（如果存在）
                previous_summary = None
                if chunk_idx > 1:
                    prev_summary_path = Path(working_dir) / f"exploration_summary_chunk{chunk_idx - 1}.md"
                    if prev_summary_path.exists():
                        previous_summary = prev_summary_path.read_text(encoding="utf-8")
                        print(f"[Phase 2.{chunk_idx}] Loaded previous summary from chunk {chunk_idx - 1} ({len(previous_summary)} chars)")

                # 构建探索 prompt - 使用工作空间路径
                prompt = self.prompt_builder.build_checkpoint_exploration_prompt(
                    file_summaries=workspace_file_summaries,
                    skills_hint=str(workspace_info["skills"]),  # 使用工作空间中的 skills 路径
                    skill_names=skill_names,
                    documented_functions=documented_functions,
                    current_state_json=json.dumps(current_state.to_dict(), indent=2, ensure_ascii=False),
                    output_dir=working_dir,
                    chunk_index=chunk_idx,
                    checkpoint_interval=self.config.checkpoint_interval,
                    previous_summary=previous_summary,
                    coverage_threshold=self.config.min_coverage_threshold,
                )

                # 在隔离工作空间中执行
                # 注意：skills_dir 传 None，因为已经复制到工作空间的 .claude/skills/
                result = self.executor.execute(
                    prompt=prompt,
                    working_dir=working_dir,
                    skills_dir=None,  # 已复制到 workspace
                    input_file=primary_input_file,
                )

                if not result.success and not result.has_trajectory:
                    print(f"[Phase 2.{chunk_idx}] Execution failed: {result.error}")
                    continue

                trajectory = self.recorder.record(result)
                print(f"[Phase 2.{chunk_idx}] Recorded {trajectory.num_steps} steps")

                # 检查 Claude 是否更新了状态
                updated_state = self._read_exploration_state(working_dir)
                if updated_state:
                    post_tested_count = updated_state.total_tested_count
                    print(f"[Phase 2.{chunk_idx}] State after execution: coverage={updated_state.coverage_ratio:.1%}, "
                          f"tested={post_tested_count}, complete={updated_state.exploration_complete}")

                    # 程序化收敛检测：检查 tested_functions 数量是否增加
                    if post_tested_count > pre_tested_count:
                        programmatic_no_progress = 0
                        print(f"[Phase 2.{chunk_idx}] Progress: tested functions increased from {pre_tested_count} to {post_tested_count}")
                    else:
                        programmatic_no_progress += 1
                        print(f"[Phase 2.{chunk_idx}] No progress: tested functions unchanged ({post_tested_count}), "
                              f"consecutive_no_progress={programmatic_no_progress}")

                    # 收敛退出检查
                    if programmatic_no_progress >= self.config.convergence_chunks:
                        print(f"[Phase 2.{chunk_idx}] Stopping: Convergence detected - "
                              f"{programmatic_no_progress} chunks without progress")
                        break

                    # 使用统一的终止条件检查（尊重用户设置的覆盖率阈值）
                    should_continue, reason = self._check_should_continue(updated_state, chunk_idx, working_dir)
                    if not should_continue:
                        print(f"[Phase 2.{chunk_idx}] Stopping: {reason}")
                        break
                else:
                    # State 读取失败也算无进展
                    programmatic_no_progress += 1
                    print(f"[Phase 2.{chunk_idx}] Warning: Failed to read state, consecutive_no_progress={programmatic_no_progress}")
                    if programmatic_no_progress >= self.config.convergence_chunks:
                        print(f"[Phase 2.{chunk_idx}] Stopping: Convergence detected - "
                              f"{programmatic_no_progress} chunks without progress")
                        break

                # 检查是否生成了探索报告
                # 只有在覆盖率达标时才因为 summary 存在而停止
                workspace_summary = Path(working_dir) / self.EXPLORATION_SUMMARY_FILE
                if workspace_summary.exists():
                    # 再次检查覆盖率是否达标
                    latest_state = self._read_exploration_state(working_dir)
                    if latest_state and latest_state.coverage_ratio >= self.config.min_coverage_threshold:
                        print(f"[Phase 2] Exploration summary generated and coverage reached {latest_state.coverage_ratio:.1%}")
                        break
                    else:
                        # 覆盖率未达标，备份 summary 文件继续探索
                        coverage_pct = latest_state.coverage_ratio if latest_state else 0
                        print(f"[Phase 2] Summary exists but coverage {coverage_pct:.1%} < {self.config.min_coverage_threshold:.1%}, backing up and continuing...")
                        # 备份当前 summary（而不是删除）
                        backup_name = f"exploration_summary_chunk{chunk_idx}.md"
                        workspace_summary.rename(Path(working_dir) / backup_name)
                        print(f"[Phase 2] Backed up summary to {backup_name}")

            # =====================================================================
            # Step 4: 复制结果到目标目录
            # =====================================================================
            print(f"\n[Phase 2] Copying results to target directory: {output_dir}")
            copied_files = copy_exploration_results(working_dir, output_dir)
            print(f"[Phase 2] Copied {len(copied_files)} files")
            for f in copied_files:
                print(f"[Phase 2]   - {f.name}")

        finally:
            # =====================================================================
            # Step 5: 清理临时工作空间
            # =====================================================================
            print(f"\n[Phase 2] Cleaning up workspace: {workspace_path}")
            if cleanup_exploration_workspace(workspace_path):
                print(f"[Phase 2] Workspace cleaned up successfully")
            else:
                print(f"[Phase 2] Warning: Failed to cleanup workspace")

        # 返回目标目录中的探索报告路径
        summary_path = Path(output_dir) / self.EXPLORATION_SUMMARY_FILE
        final_state = self._read_exploration_state(output_dir)

        # 检查是否有多个 chunk 的备份 summary（说明经过了多轮探索）
        backup_summaries = list(Path(output_dir).glob("exploration_summary_chunk*.md"))

        if backup_summaries or not summary_path.exists():
            # 多轮探索或没有 summary：使用 fallback generator 基于完整 state 重新生成
            if backup_summaries:
                print(f"[Phase 2] Found {len(backup_summaries)} backup summaries from multi-chunk exploration")
            else:
                print(f"[Phase 2] Claude did not generate summary")
            print(f"[Phase 2] Generating comprehensive summary from final state...")

            if final_state is None:
                raise SynthesisError(f"Exploration summary not generated and no state found: {summary_path}")

            try:
                fallback_summary = self._generate_fallback_exploration_summary(output_dir, final_state)
                summary_path.write_text(fallback_summary, encoding="utf-8")
                print(f"[Phase 2] Comprehensive exploration summary generated: {summary_path}")
            except Exception as e:
                raise SynthesisError(f"Failed to generate fallback summary: {e}")

        return str(summary_path)

    # =========================================================================
    # Phase 3: Task Synthesis
    # =========================================================================

    def phase3_task_synthesis(
        self,
        exploration_summary_path: str,
        file_summary: FileSummaryResult | str,
        skills_dir: str,
        output_dir: str,
        target_files: list[str] | None = None,
        task_id: str | None = None,
        isolated: bool = True,
        workspace_root: str = "./workspaces",
    ) -> HarborTask:
        """
        阶段三：任务生成

        子步骤：
        3.1 过滤文件（如果指定了 target_files）
        3.2 合成任务指令 (task_synthesize.py)
        3.3 合成引导思路 (guiding_metadata.py)
        3.4 生成 oracle 轨迹

        Args:
            exploration_summary_path: 探索报告路径
            file_summary: FileSummaryResult 对象或 JSON 文件路径
            skills_dir: 技能目录路径
            output_dir: 输出目录路径
            target_files: 用户指定的目标文件列表（None 则使用全部）
            task_id: 任务 ID（可选）
            isolated: 是否使用隔离工作空间（默认 True）
            workspace_root: 隔离工作空间根目录

        Returns:
            HarborTask 对象
        """
        import uuid
        from .utils.file_utils import setup_isolated_workspace

        print(f"\n[Phase 3] Task Synthesis")

        # 生成任务 ID
        if not task_id:
            task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # 读取探索报告
        exploration_summary = Path(exploration_summary_path).read_text(encoding="utf-8")
        print(f"[Phase 3] Loaded exploration summary: {len(exploration_summary)} chars")

        # 加载 file_summary（记录原始路径）
        file_summary_path: str | None = None
        if isinstance(file_summary, str):
            file_summary_path = file_summary
            file_summary = self.file_summarizer.load_from_json(file_summary)

        # 3.1 过滤文件
        if target_files:
            print(f"[Phase 3.1] Filtering to {len(target_files)} target files")
            file_summary = file_summary.filter_by_paths(target_files)
        else:
            print(f"[Phase 3.1] Using all {len(file_summary.files)} files")

        if not file_summary.files:
            raise SynthesisError("No files available for task synthesis")

        # ========== 创建隔离工作空间 ==========
        working_dir: str | None = None
        skills_dir_effective = skills_dir
        output_dir_effective = output_dir
        harbor_output_dir = output_dir

        if isolated:
            original_input_files = [entry.path for entry in file_summary.files]
            workspace = setup_isolated_workspace(
                task_id=task_id,
                skills_dir=skills_dir,
                input_files=original_input_files,
                exploration_summary_path=exploration_summary_path,
                file_metadata_path=file_summary_path,
                workspace_root=workspace_root,
            )

            print(f"[Phase 3] Created isolated workspace: {workspace['workspace']}")

            # 更新路径为隔离环境中的路径
            working_dir = str(workspace["workspace"])
            skills_path = workspace["skills"]
            if isinstance(skills_path, Path):
                skills_dir_effective = str(skills_path)  # .claude/skills/ contains skill subdirs directly
            output_path = workspace["output"]
            if isinstance(output_path, Path):
                output_dir_effective = str(output_path)
            harbor_path = workspace["harbor_task"]
            if isinstance(harbor_path, Path):
                harbor_output_dir = str(harbor_path)

            # 更新 file_summary 中的文件路径为隔离环境中的路径
            copied_files = workspace.get("input_files", [])
            if isinstance(copied_files, list):
                path_mapping = {
                    Path(original).name: str(copied)
                    for original, copied in zip(original_input_files, copied_files)
                }
                for entry in file_summary.files:
                    new_path = path_mapping.get(Path(entry.path).name)
                    if new_path:
                        entry.path = new_path
        # ==========================================

        # 提取可用技能
        available_skills = self._extract_skill_names(skills_dir)

        # 3.2 合成任务指令
        print(f"[Phase 3.2] Generating task instruction...")
        task_instruction = self.instruction_generator.generate_task_from_exploration(
            exploration_summary=exploration_summary,
            file_summaries=file_summary,
            skills=available_skills,
        )
        print(f"[Phase 3.2] Task instruction:")
        print("=" * 60)
        print(task_instruction)
        print("=" * 60)

        # 3.2.5 质量过滤 - 评分并可能拒绝低质量指令
        print(f"[Phase 3.2.5] Filtering task instruction quality...")
        try:
            filter_result = self.instruction_generator.filter_task_instruction(
                task_instruction=task_instruction,
                max_retries=3,
            )

            weighted_avg = filter_result.get("weighted_average", {}).get("score", 0)
            normalized_score = weighted_avg / 10.0  # Convert 0-10 to 0-1

            print(f"[Phase 3.2.5] Instruction quality score: {weighted_avg}/10 ({normalized_score:.2f})")

            # Log individual dimension scores
            for dim in ["goal_clarity", "input_clarity", "constraint_completeness",
                        "referential_clarity", "verifiability_uniqueness", "human_likeness"]:
                if dim in filter_result:
                    score = filter_result[dim].get("score", "N/A")
                    reason = filter_result[dim].get("reason", "")[:80]
                    print(f"  - {dim}: {score}/10 ({reason})")

            if normalized_score < 0.8:
                raise SynthesisError(
                    f"Task instruction quality too low: {weighted_avg}/10 (threshold: 8.0/10). "
                    f"Consider regenerating with different exploration context."
                )

            print(f"[Phase 3.2.5] Instruction passed quality filter")

        except ValueError as e:
            raise SynthesisError(f"Instruction quality filter failed: {e}")

        # 3.3 合成引导思路
        print(f"[Phase 3.3] Generating guiding metadata...")
        guiding_metadata = self.instruction_generator.generate_guiding_metadata(
            task_instruction=task_instruction,
            exploration_summary=exploration_summary,
            file_summaries=file_summary,
            skills=available_skills,
        )
        print(f"[Phase 3.3] Guiding metadata:")
        print("=" * 60)
        print(guiding_metadata)
        print("=" * 60)

        # 3.4 生成 oracle 轨迹
        print(f"[Phase 3.4] Generating oracle trajectory...")

        # 准备文件摘要
        file_summaries_dict = {entry.path: entry.summary for entry in file_summary.files}

        # 构建执行 prompt（使用任务指令 + 引导思路）
        execution_prompt = self.prompt_builder.build_goal_driven_prompt(
            task_instruction=task_instruction,
            execution_guide=guiding_metadata,
            skills_hint=skills_dir_effective,
            file_summaries=file_summaries_dict,
            output_dir=output_dir_effective,
        )

        # 使用第一个文件作为主输入
        primary_input_file = file_summary.files[0].path if file_summary.files else None

        # 执行任务
        # 注意：当使用隔离工作空间时，技能已经被复制到 .claude/skills/ 下，
        # 不需要再传 skills_dir 给 executor
        result = self.executor.execute(
            prompt=execution_prompt,
            working_dir=working_dir,
            skills_dir=None if isolated else skills_dir_effective,
            input_file=primary_input_file,
        )

        if not result.success and not result.has_trajectory:
            raise SynthesisError(f"Oracle trajectory generation failed: {result.error}")

        trajectory = self.recorder.record(result)
        print(f"[Phase 3.4] Oracle trajectory: {trajectory.num_steps} steps")

        # ========== Phase 3.4.5: PRM 轨迹验证 ==========
        if self.config.enable_prm_validation:
            print(f"[Phase 3.4.5] PRM trajectory validation...")
            prm_valid = False

            for prm_attempt in range(self.config.max_prm_retries):
                print(f"[Phase 3.4.5] PRM validation attempt {prm_attempt + 1}/{self.config.max_prm_retries}...")

                prm_result = self.trajectory_validator.validate(
                    trajectory=trajectory,
                    task_instruction=task_instruction,
                    skills_dir=skills_dir_effective,
                    working_dir=working_dir,
                )

                if prm_result.is_valid:
                    print(f"[Phase 3.4.5] PRM validation passed!")
                    prm_valid = True
                    break

                print(f"[Phase 3.4.5] PRM validation failed: {prm_result.issues}")
                print(f"[Phase 3.4.5] Feedback: {prm_result.feedback[:500]}...")

                if prm_attempt < self.config.max_prm_retries - 1:
                    # Retry oracle with PRM feedback
                    retry_prompt = self.trajectory_validator.build_retry_prompt(
                        original_prompt=execution_prompt,
                        validation_result=prm_result,
                        attempt_number=prm_attempt + 2,
                    )

                    result = self.executor.execute(
                        prompt=retry_prompt,
                        working_dir=working_dir,
                        skills_dir=None if isolated else skills_dir_effective,
                        input_file=primary_input_file,
                    )

                    if not result.success and not result.has_trajectory:
                        print(f"[Phase 3.4.5] Oracle retry {prm_attempt + 2} failed: {result.error}")
                        continue

                    trajectory = self.recorder.record(result)
                    print(f"[Phase 3.4.5] Oracle retry {prm_attempt + 2} trajectory: {trajectory.num_steps} steps")

            if not prm_valid:
                raise SynthesisError(
                    f"PRM validation failed after {self.config.max_prm_retries} attempts. "
                    f"Pipeline terminated. Issues: {prm_result.issues}"
                )
        else:
            print(f"[Phase 3.4.5] PRM validation disabled, skipping...")

        # ========== Phase 3.4.6: 提取轨迹摘要并保存最终结果 ==========
        print(f"[Phase 3.4.6] Extracting trajectory summary and saving final results...")

        # 提取轨迹摘要（截断版本，用于后续生成）
        trajectory_summary = self.trajectory_validator.extract_trajectory_for_prm(
            trajectory=trajectory,
            max_response_chars=self.config.prm_max_response_chars,
        )
        print(f"[Phase 3.4.6] Trajectory summary extracted: {len(trajectory_summary)} chars")

        # 保存最终结果到 final_res 目录
        # 清理可能存在的残留文件，防止前一个任务的输出被错误地用于测试生成
        final_res_dir = Path(working_dir) / "final_res" if working_dir else Path("/tmp/final_res")
        if final_res_dir.exists():
            import shutil as shutil_cleanup
            shutil_cleanup.rmtree(final_res_dir)
            print(f"[Phase 3.4.6] Cleaned up existing final_res directory")
        final_res_dir.mkdir(parents=True, exist_ok=True)

        # 保存轨迹摘要
        trajectory_summary_path = final_res_dir / "trajectory_summary.txt"
        trajectory_summary_path.write_text(trajectory_summary, encoding="utf-8")

        # 复制输出文件到 final_res
        final_output_files = []
        for output_file in trajectory.output_files:
            src = Path(output_file)
            if src.exists():
                import shutil
                dst = final_res_dir / src.name
                shutil.copy2(src, dst)
                final_output_files.append(str(dst))
                print(f"[Phase 3.4.6] Copied output: {src.name}")

        print(f"[Phase 3.4.6] Final results saved to: {final_res_dir}")

        # ========== Phase 3.4.7: 生成 pytest 测试（skillsbench 风格） ==========
        print(f"[Phase 3.4.7] Generating pytest tests (skillsbench style)...")

        pytest_test_path = final_res_dir / "test_outputs.py"
        pytest_content = None
        pytest_passed = False
        pytest_test_result = None

        for pytest_attempt in range(self.config.max_pytest_retries):
            print(f"[Phase 3.4.7] Pytest generation attempt {pytest_attempt + 1}/{self.config.max_pytest_retries}...")

            try:
                if pytest_attempt == 0:
                    # 首次生成
                    pytest_content = self.pytest_generator.generate(
                        task_instruction=task_instruction,
                        trajectory_summary=trajectory_summary,
                        final_files=final_output_files,
                        working_dir=str(final_res_dir),
                        test_file_path=str(pytest_test_path),
                    )
                else:
                    # 根据失败信息重新生成
                    failure_info = pytest_test_result.get_failure_summary() if pytest_test_result else "Unknown error"
                    pytest_content = self.pytest_generator.regenerate_with_feedback(
                        task_instruction=task_instruction,
                        trajectory_summary=trajectory_summary,
                        final_files=final_output_files,
                        previous_test=pytest_content,
                        failure_info=failure_info,
                        working_dir=str(final_res_dir),
                        test_file_path=str(pytest_test_path),
                    )

                # 保存测试文件
                pytest_test_path.write_text(pytest_content, encoding="utf-8")

                # 验证语法
                is_valid_syntax, syntax_error = self.pytest_generator.validate_test_syntax(pytest_content)
                if not is_valid_syntax:
                    print(f"[Phase 3.4.7] Syntax error in generated tests: {syntax_error}")
                    continue

            except Exception as e:
                print(f"[Phase 3.4.7] Test generation failed: {e}")
                continue

            # 在 final_res 目录运行 pytest 验证
            print(f"[Phase 3.4.7] Running pytest against final results...")
            pytest_test_result = self.test_executor.run_tests(
                test_file=str(pytest_test_path),
                working_dir=str(final_res_dir),
                timeout=self.config.test_timeout,
            )

            if pytest_test_result.all_passed:
                print(f"[Phase 3.4.7] Pytest passed! ({pytest_test_result.passed}/{pytest_test_result.total})")
                pytest_passed = True
                break

            print(f"[Phase 3.4.7] Pytest failed: {pytest_test_result.get_failure_summary()}")

        if not pytest_passed:
            failure_info = pytest_test_result.get_failure_summary() if pytest_test_result else "Test generation failed"
            raise SynthesisError(
                f"Pytest generation failed after {self.config.max_pytest_retries} attempts. "
                f"Pipeline terminated. Failures: {failure_info}"
            )

        # ========== Phase 3.4.6-3.4.7 (Legacy): 计算代码验证 ==========
        # 保留旧逻辑作为备用，当 pytest 生成成功时跳过
        if self.config.enable_computation_tests and not pytest_passed:
            print(f"[Phase 3.4.6] Generating computation-based tests...")

            computation_test_path = Path(working_dir) / "computation_tests.py" if working_dir else Path("/tmp/computation_tests.py")
            computation_passed = False
            comp_test_result = None

            for comp_attempt in range(self.config.max_computation_retries):
                print(f"[Phase 3.4.6] Computation test generation attempt {comp_attempt + 1}/{self.config.max_computation_retries}...")

                try:
                    if comp_attempt == 0 or comp_test_result is None:
                        # First attempt or previous attempt failed before test execution: generate new tests
                        computation_tests = self.computation_test_generator.generate(
                            trajectory=trajectory,
                            task_instruction=task_instruction,
                            input_files=[entry.path for entry in file_summary.files],
                            output_files=trajectory.output_files,
                            working_dir=working_dir,
                            skills_dir=skills_dir_effective,
                            test_file_path=str(computation_test_path),
                        )
                    else:
                        # Subsequent attempts: regenerate with failure feedback
                        computation_tests = self.computation_test_generator.regenerate(
                            trajectory=trajectory,
                            task_instruction=task_instruction,
                            failure_summary=comp_test_result.get_failure_summary(),
                            working_dir=working_dir,
                            skills_dir=skills_dir_effective,
                            test_file_path=str(computation_test_path),
                        )

                    # Save computation tests
                    computation_test_path.write_text(computation_tests, encoding="utf-8")

                    # Validate syntax
                    is_valid_syntax, syntax_error = self.computation_test_generator.validate_test_syntax(computation_tests)
                    if not is_valid_syntax:
                        print(f"[Phase 3.4.6] Syntax error in generated tests: {syntax_error}")
                        continue

                except Exception as e:
                    print(f"[Phase 3.4.6] Test generation failed: {e}")
                    continue

                # Run computation tests
                print(f"[Phase 3.4.7] Running computation tests...")
                comp_test_result = self.test_executor.run_tests(
                    test_file=str(computation_test_path),
                    working_dir=working_dir,
                    timeout=self.config.test_timeout,
                )

                # Debug output
                print(f"[Phase 3.4.7] DEBUG: passed={comp_test_result.passed}, failed={comp_test_result.failed}, "
                      f"errors={comp_test_result.errors}, total={comp_test_result.total}, "
                      f"return_code={comp_test_result.return_code}, all_passed={comp_test_result.all_passed}")

                if comp_test_result.all_passed:
                    print(f"[Phase 3.4.7] Computation tests passed! ({comp_test_result.passed}/{comp_test_result.total})")
                    computation_passed = True
                    break

                print(f"[Phase 3.4.7] Computation tests failed: {comp_test_result.get_failure_summary()}")

            if not computation_passed:
                failure_info = comp_test_result.get_failure_summary() if comp_test_result else "Test generation failed"
                raise SynthesisError(
                    f"Computation verification failed after {self.config.max_computation_retries} attempts. "
                    f"Pipeline terminated. Failures: {failure_info}"
                )
        else:
            print(f"[Phase 3.4.6-3.4.7] Computation tests disabled, skipping...")

        # 验证轨迹
        validation = self.processor.validate(trajectory)
        if not validation.is_valid:
            raise InvalidTrajectoryError(validation.errors)

        processed = self.processor.process(trajectory)

        # 准备 input_files（在 3.5.5 和打包中都需要）
        input_files = [entry.path for entry in file_summary.files]

        # ========== Phase 3.4.8: solve.sh 生成与验证（干净工作空间） ==========
        if self.config.enable_solve_verification:
            print(f"[Phase 3.4.8] Generating and validating solve.sh...")

            # solve.sh 路径
            solve_sh_path = Path(working_dir) / "solve.sh" if working_dir else Path("/tmp/solve.sh")

            # 生成初始 solve.sh（使用 trajectory_summary）
            if self.config.use_claude_for_solve and working_dir:
                print(f"[Phase 3.4.8] Using Claude Code to generate solve.sh...")
                solve_sh_content = self.solve_generator.generate_with_trajectory_summary(
                    task_instruction=task_instruction,
                    trajectory_summary=trajectory_summary,
                    input_files=input_files,
                    executor=self.executor,
                    working_dir=working_dir,
                    solve_sh_path=str(solve_sh_path),
                )
            else:
                # 使用 LLM 生成（fallback）
                solve_sh_content = self.solve_generator.generate_solve_sh(
                    trajectory=trajectory,
                    task_instruction=task_instruction,
                    expectation_tests_content=pytest_content,
                )

            # 在干净工作空间验证 solve.sh（使用 Phase 3.4.7 生成的 pytest）
            solve_passed = False
            solve_test_result = None
            verify_workspace = None

            for solve_attempt in range(self.config.max_solve_retries + 1):
                if solve_attempt > 0:
                    # 重试：根据失败信息重新生成 solve.sh
                    if self.config.use_claude_for_solve and working_dir:
                        print(f"[Phase 3.4.8] Retry {solve_attempt}/{self.config.max_solve_retries}: Refining solve.sh with Claude Code...")
                        solve_sh_content = self.solve_generator.refine_with_executor(
                            previous_solve_sh=solve_sh_content,
                            test_failures=solve_test_result.failures,
                            executor=self.executor,
                            working_dir=working_dir,
                            solve_sh_path=str(solve_sh_path),
                            task_instruction=task_instruction,
                            trajectory_summary=trajectory_summary,
                        )
                    else:
                        print(f"[Phase 3.4.8] Retry {solve_attempt}/{self.config.max_solve_retries}: Regenerating solve.sh with LLM...")
                        solve_sh_content = self.solve_generator.regenerate_with_llm(
                            trajectory=trajectory,
                            task_instruction=task_instruction,
                            test_failures=solve_test_result.failures,
                            previous_solve_sh=solve_sh_content,
                            expectation_tests_content=pytest_content,
                        )

                # 在干净工作空间验证
                print(f"[Phase 3.4.8] Verifying solve.sh in clean workspace...")
                solve_passed, solve_test_result, verify_workspace = self.solve_verifier.verify_in_clean_workspace(
                    solve_sh_content=solve_sh_content,
                    test_file_content=pytest_content,
                    input_files=input_files,
                    skills_dir=skills_dir_effective,
                    timeout=self.config.solve_timeout,
                    conda_env=self.config.conda_env,
                    cleanup=False,  # 保留工作空间用于调试
                )

                if solve_passed:
                    print(f"[Phase 3.4.8] solve.sh validation passed!")
                    break

                print(f"[Phase 3.4.8] solve.sh validation failed: {solve_test_result.get_failure_summary()}")

            if not solve_passed:
                print(f"[Phase 3.4.8] ERROR: solve.sh still failing after {self.config.max_solve_retries} retries")
                raise SolveSynthesisError(
                    f"solve.sh validation failed after {self.config.max_solve_retries} retries: "
                    f"{solve_test_result.get_failure_summary()}"
                )
        else:
            # 禁用验证时，仍生成 solve.sh 但不验证
            solve_sh_content = self.solve_generator.generate_from_trajectory(trajectory)
        # ================================================

        # ========== Phase 3.4.9: 打包（skillsbench 格式） ==========
        print(f"[Phase 3.4.9] Packaging task in skillsbench format...")

        # 路径规范化
        skillsbench_normalizer = SkillsbenchPathNormalizer(
            workspace_path=working_dir,
            input_files=input_files,
        )
        # 规范化所有文件的路径：instruction、pytest、solve.sh
        instruction_normalized = skillsbench_normalizer.normalize(task_instruction)
        pytest_content_normalized = skillsbench_normalizer.normalize_tests(pytest_content)
        solve_sh_normalized = skillsbench_normalizer.normalize_solve_sh(solve_sh_content)

        # 使用 skillsbench 格式打包
        task = self.packager.package_skillsbench_format(
            task_id=task_id,
            instruction=instruction_normalized,
            tests_content=pytest_content_normalized,
            solve_sh_content=solve_sh_normalized,
            input_files=input_files,
            skills_dir=skills_dir_effective,
            output_dir=harbor_output_dir,
            metadata={
                "difficulty": self.config.default_difficulty,
                "category": self.config.default_category,
            },
            pip_packages=processed.pip_packages,
            used_skills=processed.used_skills,
            output_files=trajectory.output_files,
        )

        # 保存引导思路到任务目录
        guiding_path = Path(task.task_path) / "guiding_metadata.md"
        guiding_path.write_text(guiding_metadata, encoding="utf-8")
        print(f"[Phase 3] Guiding metadata saved to: {guiding_path}")

        # 移动中间产物文件到单独目录，保持 harbor 任务目录干净
        if isolated and working_dir:
            self._move_synthesis_artifacts(
                task_path=task.task_path,
                workspace_dir=working_dir,
            )

        print(f"\n[Phase 3] Task generated: {task.task_path}")
        if isolated:
            print(f"[Phase 3] Workspace: {working_dir}")
        return task

    def phase3_batch_task_synthesis(
        self,
        exploration_summary_path: str,
        file_summary: FileSummaryResult | str,
        skills_dir: str,
        output_dir: str,
        target_file_groups: list[list[str]] | None = None,
        isolated: bool = True,
        workspace_root: str = "./workspaces",
    ) -> list[HarborTask]:
        """
        阶段三（批量）：并发生成多个任务

        Args:
            exploration_summary_path: 探索报告路径
            file_summary: FileSummaryResult 对象或 JSON 文件路径
            skills_dir: 技能目录路径
            output_dir: 输出目录路径
            target_file_groups: 目标文件组列表，每组生成一个任务
                如果为 None，则为每个文件单独生成一个任务
            isolated: 是否使用隔离工作空间（默认 True）
            workspace_root: 隔离工作空间根目录

        Returns:
            生成的 HarborTask 列表
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm

        print(f"\n[Phase 3 Batch] Task Synthesis")
        if isolated:
            print(f"[Phase 3 Batch] Isolated mode enabled, workspace root: {workspace_root}")

        # 加载 file_summary
        if isinstance(file_summary, str):
            file_summary = self.file_summarizer.load_from_json(file_summary)

        # 确定要处理的文件组
        if target_file_groups is None:
            # 默认：每个文件一个任务
            target_file_groups = [[entry.path] for entry in file_summary.files]

        print(f"[Phase 3 Batch] Generating {len(target_file_groups)} tasks (workers={self.config.max_workers})")

        tasks: list[HarborTask] = []
        errors: list[tuple[int, str]] = []

        def generate_single_task(idx: int, target_files: list[str]) -> tuple[int, HarborTask | None, str | None]:
            """Generate a single task"""
            try:
                task_id = f"task_{idx:04d}_{datetime.now().strftime('%H%M%S')}"
                task = self.phase3_task_synthesis(
                    exploration_summary_path=exploration_summary_path,
                    file_summary=file_summary,
                    skills_dir=skills_dir,
                    output_dir=output_dir,
                    target_files=target_files,
                    task_id=task_id,
                    isolated=isolated,
                    workspace_root=workspace_root,
                )
                return (idx, task, None)
            except Exception as e:
                return (idx, None, str(e))

        # 并发生成任务
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_idx = {
                executor.submit(generate_single_task, i, group): i
                for i, group in enumerate(target_file_groups)
            }

            pbar = tqdm(
                as_completed(future_to_idx),
                total=len(target_file_groups),
                desc="Generating tasks",
                disable=not self.config.show_progress,
            )

            for future in pbar:
                idx = future_to_idx[future]
                try:
                    result_idx, task, error = future.result()
                    if task:
                        tasks.append(task)
                        pbar.set_postfix_str(f"Task {result_idx} OK")
                    else:
                        errors.append((result_idx, error or "Unknown error"))
                        pbar.set_postfix_str(f"Task {result_idx} FAILED")
                except Exception as e:
                    errors.append((idx, str(e)))
                    pbar.set_postfix_str(f"Task {idx} ERROR")

        print(f"\n[Phase 3 Batch] Generated {len(tasks)} tasks, {len(errors)} errors")
        if errors:
            for idx, err in errors:
                print(f"[Phase 3 Batch] Task {idx} failed: {err}")

        return tasks

    def _read_exploration_state(self, output_dir: str) -> ExplorationState | None:
        """
        从文件读取探索状态。

        Args:
            output_dir: 输出目录路径

        Returns:
            ExplorationState 对象，如果文件不存在则返回 None
        """
        state_path = Path(output_dir) / self.EXPLORATION_STATE_FILE
        if not state_path.exists():
            return None
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            return ExplorationState.from_dict(data)
        except Exception as e:
            print(f"[Pipeline] Warning: Failed to read exploration state: {e}")
            return None

    def _save_exploration_state(self, state: ExplorationState, output_dir: str) -> None:
        """
        保存探索状态到文件。

        Args:
            state: ExplorationState 对象
            output_dir: 输出目录路径
        """
        state_path = Path(output_dir) / self.EXPLORATION_STATE_FILE
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps(state.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _read_checkpoint_files(self, output_dir: str) -> str:
        """
        读取所有 checkpoint 文件的内容。

        Args:
            output_dir: 输出目录路径

        Returns:
            所有 checkpoint 文件内容的拼接字符串
        """
        checkpoint_contents = []
        output_path = Path(output_dir)

        # 按编号顺序读取 checkpoint 文件
        checkpoint_files = sorted(output_path.glob("checkpoint_*.md"))
        for cp_file in checkpoint_files:
            try:
                content = cp_file.read_text(encoding="utf-8")
                checkpoint_contents.append(f"### {cp_file.name}\n\n{content}")
            except Exception as e:
                checkpoint_contents.append(f"### {cp_file.name}\n\n[Error reading: {e}]")

        if not checkpoint_contents:
            return "[No checkpoint files found]"

        return "\n\n---\n\n".join(checkpoint_contents)

    def _generate_fallback_exploration_summary(
        self,
        output_dir: str,
        exploration_state: ExplorationState,
    ) -> str:
        """
        当 Claude 未生成探索摘要时，使用 LLM 从 checkpoint 和状态生成回退摘要。

        Args:
            output_dir: 输出目录路径
            exploration_state: 当前探索状态

        Returns:
            生成的探索摘要内容
        """
        from .execution.prompt_builder import FALLBACK_SUMMARY_GENERATION_PROMPT

        # 读取所有 checkpoint 文件
        checkpoint_content = self._read_checkpoint_files(output_dir)

        # 构建提示
        prompt = FALLBACK_SUMMARY_GENERATION_PROMPT.format(
            exploration_state_json=json.dumps(exploration_state.to_dict(), indent=2, ensure_ascii=False),
            checkpoint_files_content=checkpoint_content,
        )

        # 使用 LLM 生成摘要
        summary = self.llm_client_synthesis.generate(
            system_prompt="You are an expert at summarizing code exploration results. Generate clear, structured documentation.",
            user_prompt=prompt,
            temperature=0.3,
        )

        return summary

    def _check_should_continue(
        self,
        state: ExplorationState,
        chunk_count: int,
        output_dir: str | None = None,
    ) -> tuple[bool, str]:
        """
        检查是否应该继续探索。

        Args:
            state: 当前探索状态
            chunk_count: 当前 chunk 计数
            output_dir: 输出目录（用于重置状态）

        Returns:
            (should_continue, reason) 元组
        """
        # 条件 1: 覆盖率达标（最高优先级，用户明确指定的目标）
        if state.coverage_ratio >= self.config.min_coverage_threshold:
            return False, f"Coverage threshold reached: {state.coverage_ratio:.1%}"

        # 条件 2: Claude 主动标记完成（只有在覆盖率达标时才尊重）
        # 如果覆盖率未达标但 Claude 标记完成，继续探索并打印警告
        if state.exploration_complete:
            print(f"[Warning] Claude marked complete at {state.coverage_ratio:.1%} coverage, "
                  f"but target is {self.config.min_coverage_threshold:.1%}. Continuing exploration...")
            # 重置 exploration_complete 状态，让下一个 chunk 继续探索
            if output_dir:
                state.exploration_complete = False
                state.completion_reason = f"Reset: target coverage is {self.config.min_coverage_threshold:.1%}"
                self._save_exploration_state(state, output_dir)
                print(f"[Warning] Reset exploration_complete=False for next chunk")
            # 不返回，继续检查其他条件

        # 条件 3: 收敛检测（连续多个 chunk 无进展则放弃）
        if state.consecutive_no_progress >= self.config.convergence_chunks:
            return False, f"Convergence detected: {state.consecutive_no_progress} chunks without progress"

        # 条件 4: 安全阀
        if chunk_count >= self.config.max_exploration_chunks:
            return False, f"Safety limit reached: {chunk_count} chunks"

        return True, "Continue exploration"

    def _extract_documented_functions(self, skills_dir: str | None) -> dict[str, list[str]]:
        """
        从 SKILL.md 文件中提取文档化的函数/操作列表，按 skill 分组。

        提取策略：
        1. #### 标题：如 "#### Merge PDFs" → "Merge PDFs"（操作描述）
        2. 代码块中的函数定义：def func_name()
        3. 代码块中的类/函数调用：ClassName(), module.func()
        4. 内联代码格式：`func_name()`

        排除：## 大标题（如 Overview, Quick Start 等章节标题）

        Args:
            skills_dir: 技能目录路径

        Returns:
            skill 名称到函数列表的映射，例如 {"pdf": ["Merge PDFs", "PdfReader"], "excel": ["read_excel"]}
        """
        import re

        if not skills_dir:
            return {}

        functions_by_skill: dict[str, list[str]] = {}
        skills_path = Path(skills_dir)
        if not skills_path.exists():
            return {}

        # 常见的章节标题词，需要排除
        section_keywords = {
            "overview", "introduction", "quick", "start", "getting", "started",
            "installation", "setup", "usage", "example", "examples", "reference",
            "python", "command", "line", "tools", "common", "next", "steps",
            "advanced", "basic", "summary", "conclusion", "requirements",
            "dependencies", "configuration", "config", "settings", "notes",
            "troubleshooting", "faq", "api", "methods", "functions", "classes",
        }

        # 遍历每个技能目录
        for skill_dir in skills_path.iterdir():
            if not skill_dir.is_dir() or skill_dir.name.startswith('.'):
                continue

            skill_name = skill_dir.name
            skill_funcs: list[str] = []

            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue

            try:
                content = skill_md.read_text(encoding="utf-8")

                # 1. 匹配 #### 标题（操作描述，如 "#### Merge PDFs"）
                operation_headers = re.findall(r'^####\s+(.+)$', content, re.MULTILINE)
                for header in operation_headers:
                    header = header.strip()
                    if header and header.lower() not in section_keywords:
                        skill_funcs.append(header)

                # 2. 匹配代码块中的函数定义
                code_funcs = re.findall(r'def\s+(\w+)\s*\(', content)
                skill_funcs.extend(code_funcs)

                # 3. 匹配代码块中的类实例化和方法调用（如 PdfReader(), page.extract_text()）
                # 匹配 ClassName(...)
                class_calls = re.findall(r'\b([A-Z][a-zA-Z0-9]+)\s*\(', content)
                skill_funcs.extend(class_calls)
                # 匹配 module.func(...) 或 obj.method(...)
                method_calls = re.findall(r'\.([a-z_][a-z0-9_]*)\s*\(', content, re.IGNORECASE)
                # 过滤常见的非函数名
                non_funcs = {"pdf", "pages", "page", "open", "write", "read", "close", "append", "extend"}
                method_calls = [m for m in method_calls if m.lower() not in non_funcs and len(m) > 2]
                skill_funcs.extend(method_calls)

                # 4. 匹配 `func_name()` 格式的内联代码
                inline_funcs = re.findall(r'`(\w+)\(\)`', content)
                skill_funcs.extend(inline_funcs)

                # 去重并过滤
                seen = set()
                filtered_funcs = []
                for func in skill_funcs:
                    func_lower = func.lower()
                    if func_lower not in seen and func_lower not in section_keywords:
                        seen.add(func_lower)
                        filtered_funcs.append(func)

                if filtered_funcs:
                    functions_by_skill[skill_name] = filtered_funcs

            except Exception as e:
                print(f"[Pipeline] Warning: Failed to parse {skill_md}: {e}")

        return functions_by_skill

    def _read_output_files(self, output_files: list[str], max_size: int = 10000) -> dict[str, str]:
        """
        读取输出文件内容。

        Args:
            output_files: 输出文件路径列表
            max_size: 每个文件最大读取字节数

        Returns:
            文件路径到内容的映射
        """
        contents = {}
        skip_patterns = ['/dev/', '/tmp/', '/proc/', '/sys/']

        for file_path in output_files:
            # 跳过特殊路径
            if any(file_path.startswith(p) for p in skip_patterns):
                continue

            try:
                p = Path(file_path)
                if p.exists() and p.is_file():
                    content = p.read_text(encoding="utf-8", errors="ignore")
                    if len(content) > max_size:
                        content = content[:max_size] + f"\n... [truncated, total {len(content)} chars]"
                    contents[file_path] = content
            except Exception as e:
                contents[file_path] = f"[Error reading file: {e}]"

        return contents

    def _build_retry_prompt_with_test_feedback(
        self,
        original_prompt: str,
        failure_history: list[dict],
        current_retry: int,
    ) -> str:
        """
        构建带累积测试失败反馈的重试 prompt。

        累积所有历史失败信息，让模型在每次重试时都能看到所有之前的失败，
        避免修复一个错误时重新引入之前已修复的错误。

        Args:
            original_prompt: 原始执行 prompt
            failure_history: 累积的失败历史 [{"attempt": 0, "failures": [...], "error": optional}, ...]
            current_retry: 当前重试编号 (1-based)

        Returns:
            增强后的重试 prompt
        """
        feedback_lines = [
            "\n",
            "=" * 60,
            "## IMPORTANT: Previous Attempts Failed Verification Tests",
            "=" * 60,
            "",
        ]

        # 遍历所有历史失败
        for record in failure_history:
            attempt = record["attempt"]
            failures = record.get("failures", [])
            error = record.get("error")

            if attempt == 0:
                feedback_lines.append("### Attempt 1 (Initial Execution):")
            else:
                feedback_lines.append(f"### Attempt {attempt + 1} (Retry {attempt}):")
            feedback_lines.append("")

            # 如果是执行错误（非测试失败）
            if error:
                feedback_lines.append(f"   Execution failed: {error}")
                feedback_lines.append("")
                continue

            # 显示测试失败
            if not failures:
                feedback_lines.append("   No test failures recorded.")
                feedback_lines.append("")
                continue

            for i, failure in enumerate(failures, 1):
                test_name = getattr(failure, 'test_name', str(failure))
                error_msg = getattr(failure, 'error_message', '')
                feedback_lines.append(f"{i}. **{test_name}**")
                if error_msg:
                    feedback_lines.append(f"   Error: {error_msg}")
                feedback_lines.append("")

        feedback_lines.extend([
            "---",
            f"This is retry {current_retry} of {self.config.max_verification_retries}.",
            "",
            "Please carefully review ALL the errors above and ensure your next attempt:",
            "1. **Avoids ALL previously encountered errors** (not just the most recent ones)",
            "2. Creates all required output files in the correct locations",
            "3. Uses the correct output format (JSON, CSV, etc.) and field names",
            "4. Follows the exact specifications in the instruction",
            "",
            "Now, please re-execute the task correctly.",
            "=" * 60,
            "",
        ])

        feedback_text = "\n".join(feedback_lines)
        return original_prompt + feedback_text

    def _move_synthesis_artifacts(
        self,
        task_path: str,
        workspace_dir: str,
    ) -> None:
        """
        将合成过程中的中间产物文件从 Harbor 任务目录移到工作空间的单独目录。

        这样可以保持最终的 Harbor 任务目录干净，只包含运行任务所需的文件。

        Args:
            task_path: Harbor 任务目录路径 (如 ./workspaces/task_xxx/harbor_task/task_xxx)
            workspace_dir: 工作空间根目录 (如 ./workspaces/task_xxx)
        """
        from .utils.file_utils import move_file, ensure_directory

        task_dir = Path(task_path)
        workspace = Path(workspace_dir)

        # 创建中间产物存放目录
        artifacts_dir = workspace / "synthesis_artifacts"
        ensure_directory(artifacts_dir)

        # 需要移动的文件列表
        artifacts_to_move = [
            # (源路径相对于 task_dir, 目标文件名)
            ("trajectory.json", "trajectory.json"),
            ("guiding_metadata.md", "guiding_metadata.md"),
            ("environment/extracted_dependencies.json", "extracted_dependencies.json"),
        ]

        moved_count = 0
        for rel_src, dst_name in artifacts_to_move:
            src_path = task_dir / rel_src
            if src_path.exists():
                dst_path = artifacts_dir / dst_name
                result = move_file(src_path, dst_path)
                if result:
                    moved_count += 1
                    print(f"[Cleanup] Moved {rel_src} -> synthesis_artifacts/{dst_name}")

        if moved_count > 0:
            print(f"[Cleanup] Moved {moved_count} artifact(s) to {artifacts_dir}")

    def _read_exploration_summary(self, output_dir: str) -> str:
        """
        从固定文件读取探索总结。

        Args:
            output_dir: 输出目录路径

        Returns:
            总结文件内容，如果文件不存在则返回空字符串
        """
        summary_path = Path(output_dir) / self.EXPLORATION_SUMMARY_FILE
        if summary_path.exists():
            return summary_path.read_text(encoding="utf-8")
        return ""

    def _cleanup_exploration_artifacts(
        self,
        trajectory: Trajectory | None,
        output_dir: str,
        input_files: list[str],
    ) -> None:
        """
        彻底清理探索阶段创建的所有文件和目录。

        策略：
        1. 删除 trajectory.output_files 中的文件
        2. 删除 output_dir 下除 exploration_summary.md 外的所有文件
        3. 删除探索阶段创建的脚本目录（如 scripts/）
        4. 保护输入文件不被删除

        Args:
            trajectory: 探索轨迹
            output_dir: 输出目录路径
            input_files: 输入文件列表（需要保护）
        """
        import shutil

        skip_patterns = ['/dev/', '/tmp/', '/proc/', '/sys/', '/usr/', '/bin/', '/lib/']
        protected_files = set(input_files)  # 输入文件不能删除
        protected_files.add(str(Path(output_dir) / self.EXPLORATION_SUMMARY_FILE))

        removed_count = 0
        errors = []

        # 1. 删除 trajectory.output_files
        if trajectory and trajectory.output_files:
            for f in trajectory.output_files:
                if any(f.startswith(p) for p in skip_patterns):
                    continue
                if f in protected_files:
                    continue
                try:
                    p = Path(f)
                    if p.exists():
                        if p.is_file():
                            p.unlink()
                            removed_count += 1
                        elif p.is_dir():
                            shutil.rmtree(p)
                            removed_count += 1
                except Exception as e:
                    errors.append(f"{f}: {e}")

        # 2. 清理 output_dir 下的探索产物（除了 exploration_summary.md）
        output_path = Path(output_dir)
        if output_path.exists():
            for item in output_path.iterdir():
                if str(item) in protected_files:
                    continue
                if item.name == self.EXPLORATION_SUMMARY_FILE:
                    continue
                try:
                    if item.is_file():
                        item.unlink()
                        removed_count += 1
                    elif item.is_dir():
                        shutil.rmtree(item)
                        removed_count += 1
                except Exception as e:
                    errors.append(f"{item}: {e}")

        # 3. 删除可能在工作目录下创建的脚本目录
        working_dir = Path(input_files[0]).parent if input_files else None
        if working_dir and working_dir.exists():
            common_script_dirs = ['scripts', 'helpers', 'utils', '__pycache__']
            for dir_name in common_script_dirs:
                script_dir = working_dir / dir_name
                if script_dir.exists() and script_dir.is_dir():
                    try:
                        shutil.rmtree(script_dir)
                        removed_count += 1
                    except Exception as e:
                        errors.append(f"{script_dir}: {e}")

        print(f"[Phase 1.6] Cleaned {removed_count} exploration artifacts")
        if errors:
            for err in errors[:5]:  # 最多显示5个错误
                print(f"[Phase 1.6] Warning: Could not remove {err}")

    def _extract_skill_names(self, skills_dir: str | None) -> list[str]:
        """
        从 skills 目录提取可用技能名称。

        Args:
            skills_dir: Path to skills directory

        Returns:
            List of skill names found in the directory
        """
        if not skills_dir:
            return []
        skills_path = Path(skills_dir)
        if not skills_path.exists():
            return []
        return [d.name for d in skills_path.iterdir() if d.is_dir() and not d.name.startswith('.')]

    def _save_raw_trajectory(
        self,
        result: "ExecutionResult",
        trajectory: "Trajectory",
        output_dir: str,
        task_id: str | None = None,
    ) -> str:
        """
        Save raw trajectory data before processing for debugging comparison.

        Args:
            result: ExecutionResult from executor
            trajectory: Recorded trajectory object
            output_dir: Output directory
            task_id: Optional task ID for filename

        Returns:
            Path to saved file
        """
        from .execution.claude_executor import ExecutionResult

        # Create debug directory
        debug_dir = Path(output_dir) / "_debug_trajectories"
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"raw_trajectory_{task_id or timestamp}.json"
        filepath = debug_dir / filename

        # Prepare raw data for comparison
        raw_data = {
            "saved_at": datetime.now().isoformat(),
            "task_id": task_id,
            # Original execution result data
            "execution_result": {
                "success": result.success,
                "exit_code": result.exit_code,
                "duration_ms": result.duration_ms,
                "working_dir": result.working_dir,
                "error": result.error,
                "has_trajectory": result.has_trajectory,
                "raw_events_count": len(result.trajectory_data.get("events", [])) if result.trajectory_data else 0,
            },
            # Raw events from Claude Code (unprocessed)
            "raw_trajectory_data": result.trajectory_data,
            # Recorded trajectory summary (after TrajectoryRecorder processing)
            "recorded_trajectory_summary": {
                "session_id": trajectory.session_id,
                "model": trajectory.model,
                "success": trajectory.success,
                "duration_ms": trajectory.duration_ms,
                "num_steps": trajectory.num_steps,
                "num_tool_use_steps": len(trajectory.tool_use_steps),
                "input_files": trajectory.input_files,
                "output_files": trajectory.output_files,
                "edited_files": trajectory.edited_files,
                "notebook_files": trajectory.notebook_files,
                "used_skills": trajectory.used_skills,
                "bash_commands": trajectory.bash_commands,
                "web_urls": trajectory.web_urls,
                "search_queries": trajectory.search_queries,
                "glob_patterns": trajectory.glob_patterns,
                "grep_patterns": trajectory.grep_patterns,
            },
            # Detailed steps for comparison
            "recorded_steps": [
                {
                    "step_id": step.step_id,
                    "action_type": step.action_type,
                    "tool_name": step.tool_name,
                    "tool_input": step.tool_input,
                    "timestamp": step.timestamp,
                    "reasoning": step.reasoning[:200] if step.reasoning else None,
                }
                for step in trajectory.steps
            ],
        }

        # Write to file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, indent=2, ensure_ascii=False)

        return str(filepath)

    def _generate_file_summary(
        self,
        file_path: str,
        max_chars: int = 50,
        max_read_lines: int = 2000,
    ) -> str:
        """
        Generate summary for a single file using Claude Code CLI.

        Args:
            file_path: Path to the file
            max_chars: Maximum characters for the summary
            max_read_lines: Maximum lines to read from large files

        Returns:
            File summary string
        """
        return self.executor.generate_file_summary(
            file_path=file_path,
            max_chars=max_chars,
            max_read_lines=max_read_lines,
        )

    def synthesize(
        self,
        input_file: str | None = None,
        input_files: list[str] | None = None,
        skills_dir: str | None = None,
        prompt_style: str = "minimal",
        max_steps: int = 10,
        domain: str | None = None,
        output_dir: str = "./workspaces",
        task_id: str | None = None,
        skip_summary: bool = False,
        no_goal_driven: bool = False,
    ) -> HarborTask:
        """
        Run the complete synthesis pipeline with adaptive exploration.

        The exploration phase uses checkpoint-driven adaptive exploration that
        automatically determines when to stop based on:
        - Coverage threshold (default 90%)
        - Convergence detection (3 consecutive chunks without progress)
        - Safety limit (max chunks)
        - Claude's explicit completion signal

        Args:
            input_file: Path to single input file (backward compatible)
            input_files: List of input file paths (supports multiple files)
            skills_dir: Path to skills directory
            prompt_style: Prompt template style
            max_steps: Maximum exploration steps per chunk
            domain: Optional domain hint
            output_dir: Output directory for generated tasks
            task_id: Optional task ID (generated if None)
            skip_summary: If True, skip Phase 0 summary generation
            no_goal_driven: If True, skip Goal-driven execution

        Returns:
            HarborTask with generated task information

        Raises:
            InvalidTrajectoryError: If trajectory fails validation
            SynthesisError: If synthesis fails
        """
        # Normalize input files (backward compatibility)
        if input_files is None:
            input_files = [input_file] if input_file else []
        elif input_file and input_file not in input_files:
            input_files = [input_file] + input_files

        print(f"[Pipeline] Starting synthesis...")
        print(f"[Pipeline] Input files: {input_files}")
        print(f"[Pipeline] Skills dir: {skills_dir}")
        print(f"[Pipeline] Prompt style: {prompt_style}")
        print(f"[Pipeline] Skip summary: {skip_summary}")

        # Phase 0: File Summary Generation
        file_summaries = {}
        if input_files and not skip_summary:
            print("\n[Phase 0] Generating file summaries...")
            for file_path in input_files:
                print(f"[Phase 0] Summarizing: {file_path}")
                try:
                    summary = self._generate_file_summary(file_path)
                    file_summaries[file_path] = summary
                    print(f"[Phase 0] Summary: {summary}")
                except Exception as e:
                    print(f"[Phase 0] Warning: Failed to generate summary for {file_path}: {e}")
                    file_summaries[file_path] = f"[摘要生成失败，请直接读取文件 {file_path}]"
            print(f"[Phase 0] Generated {len(file_summaries)} file summaries")

        # Phase 1: 自适应探索（检查点驱动）
        enable_goal_driven = self.config.enable_goal_driven and not no_goal_driven
        max_chunks = self.config.max_exploration_chunks
        print(f"\n[Phase 1] Starting adaptive exploration (max_chunks={max_chunks}, goal_driven={enable_goal_driven})...")

        prompt_config = PromptConfig(
            template_style=PromptStyle(prompt_style),
            max_steps=max_steps,
        )
        self.prompt_builder = PromptBuilder(prompt_config)

        # Use first input file for working directory setup (executor copies one file)
        primary_input_file = input_files[0] if input_files else None

        trajectory = None
        result = None

        # 必须有 file_summaries
        if not file_summaries:
            raise SynthesisError("file_summaries is required for exploration. Use --skip-summary=False or provide input files.")

        # [FIX] 将 output_dir 转换为绝对路径，确保 Claude 和 Pipeline 使用一致的路径
        output_dir = str(Path(output_dir).resolve())
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print(f"[Phase 1] Exploration output dir: {output_dir}")

        # 提取技能名称
        skill_names = self._extract_skill_names(skills_dir)
        documented_functions = self._extract_documented_functions(skills_dir)
        total_funcs = sum(len(funcs) for funcs in documented_functions.values())
        print(f"[Phase 1] Skills: {skill_names}, Documented functions: {total_funcs}")
        for skill, funcs in documented_functions.items():
            print(f"[Phase 1]   - {skill}: {len(funcs)} functions")

        # 初始化探索状态（支持多 skills）
        initial_state = ExplorationState(
            skill_names=skill_names,
            session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            started_at=datetime.now().isoformat(),
            documented_functions=documented_functions,
        )
        self._save_exploration_state(initial_state, output_dir)
        print(f"[Phase 1] Initial state saved: {output_dir}/{self.EXPLORATION_STATE_FILE}")

        # 自适应探索循环
        # 程序化收敛检测（不依赖 Claude 更新 consecutive_no_progress）
        programmatic_no_progress = 0
        chunk_idx = 0

        while chunk_idx < max_chunks:
            chunk_idx += 1
            print(f"\n[Phase 1.{chunk_idx}] Exploration chunk {chunk_idx}/{max_chunks}")

            # 读取当前探索状态
            current_state = self._read_exploration_state(output_dir)
            if current_state is None:
                current_state = initial_state

            # 记录执行前的 tested_functions 数量
            pre_tested_count = current_state.total_tested_count

            # 检查是否应该继续
            should_continue, reason = self._check_should_continue(current_state, chunk_idx - 1, output_dir)
            if not should_continue:
                print(f"[Phase 1.{chunk_idx}] Stopping exploration: {reason}")
                break

            # 程序化收敛退出检查
            if programmatic_no_progress >= self.config.convergence_chunks:
                print(f"[Phase 1.{chunk_idx}] Stopping: Convergence detected - "
                      f"{programmatic_no_progress} chunks without progress")
                break

            print(f"[Phase 1.{chunk_idx}] Current coverage: {current_state.coverage_ratio:.1%}")
            print(f"[Phase 1.{chunk_idx}] Tested functions: {current_state.total_tested_count}/{current_state.total_documented_count}"
                  f" (no_progress={programmatic_no_progress})")

            # 读取上一个 chunk 的 summary 备份（如果存在）
            previous_summary = None
            if chunk_idx > 1:
                prev_summary_path = Path(output_dir) / f"exploration_summary_chunk{chunk_idx - 1}.md"
                if prev_summary_path.exists():
                    previous_summary = prev_summary_path.read_text(encoding="utf-8")
                    print(f"[Phase 1.{chunk_idx}] Loaded previous summary from chunk {chunk_idx - 1} ({len(previous_summary)} chars)")

            # 构建检查点驱动的探索 prompt（支持多 skills）
            prompt = self.prompt_builder.build_checkpoint_exploration_prompt(
                file_summaries=file_summaries,
                skills_hint=skills_dir,
                skill_names=skill_names,
                documented_functions=documented_functions,
                current_state_json=json.dumps(current_state.to_dict(), indent=2, ensure_ascii=False),
                output_dir=output_dir,
                chunk_index=chunk_idx,
                checkpoint_interval=self.config.checkpoint_interval,
                previous_summary=previous_summary,
                coverage_threshold=self.config.min_coverage_threshold,
            )

            print(f"[Phase 1.{chunk_idx}] Prompt length: {len(prompt)} chars")

            # 执行探索（带重试）
            # [FIX] 传递 working_dir 确保状态文件写入正确位置
            max_retries = 3
            for attempt in range(max_retries):
                result = self.executor.execute(
                    prompt=prompt,
                    working_dir=output_dir,  # 关键修复：传递工作目录
                    skills_dir=skills_dir,
                    input_file=primary_input_file,
                )

                if not result.success and not result.has_trajectory:
                    print(f"[Phase 1.{chunk_idx}] Attempt {attempt + 1} failed: {result.error or 'Unknown error'}")
                    if attempt < max_retries - 1:
                        print(f"[Phase 1.{chunk_idx}] Retrying...")
                        continue
                    raise SynthesisError(f"Execution failed after {max_retries} attempts: {result.error or 'Unknown error'}")

                trajectory = self.recorder.record(result)
                print(f"[Phase 1.{chunk_idx}] Recorded {trajectory.num_steps} steps")

                # Check if we got meaningful output
                if trajectory.num_steps > 1:
                    break  # Success
                elif attempt < max_retries - 1:
                    print(f"[Phase 1.{chunk_idx}] Attempt {attempt + 1}: Only {trajectory.num_steps} steps. Retrying...")
                else:
                    print(f"[Phase 1.{chunk_idx}] Warning: Low step count after {max_retries} attempts")

            # **保存本次 chunk 的探索轨迹**
            chunk_trajectory_path = self._save_raw_trajectory(
                result=result,
                trajectory=trajectory,
                output_dir=output_dir,
                task_id=f"{task_id or 'task'}_exploration_chunk_{chunk_idx}",
            )
            print(f"[Phase 1.{chunk_idx}] Trajectory saved: {chunk_trajectory_path}")

            # 从轨迹中提取本次 chunk 的信息
            chunk_skills = self.recorder.get_used_skills(trajectory)
            chunk_written_files = trajectory.output_files

            print(f"[Phase 1.{chunk_idx}] Skills used: {chunk_skills}")
            print(f"[Phase 1.{chunk_idx}] Files written: {chunk_written_files}")

            # 检查 exploration_state.json 是否已更新
            updated_state = self._read_exploration_state(output_dir)
            if updated_state:
                post_tested_count = updated_state.total_tested_count
                print(f"[Phase 1.{chunk_idx}] State updated: coverage={updated_state.coverage_ratio:.1%}, "
                      f"tested={post_tested_count}, complete={updated_state.exploration_complete}")

                # 程序化收敛检测：检查 tested_functions 数量是否增加
                if post_tested_count > pre_tested_count:
                    programmatic_no_progress = 0
                    print(f"[Phase 1.{chunk_idx}] Progress: tested functions increased from {pre_tested_count} to {post_tested_count}")
                else:
                    programmatic_no_progress += 1
                    print(f"[Phase 1.{chunk_idx}] No progress: tested functions unchanged ({post_tested_count}), "
                          f"consecutive_no_progress={programmatic_no_progress}")

                # 收敛退出检查
                if programmatic_no_progress >= self.config.convergence_chunks:
                    print(f"[Phase 1.{chunk_idx}] Stopping: Convergence detected - "
                          f"{programmatic_no_progress} chunks without progress")
                    break

                # 使用统一的终止条件检查（尊重用户设置的覆盖率阈值）
                should_continue, reason = self._check_should_continue(updated_state, chunk_idx, output_dir)
                if not should_continue:
                    print(f"[Phase 1.{chunk_idx}] Stopping exploration: {reason}")
                    break
            else:
                # State 读取失败也算无进展
                programmatic_no_progress += 1
                print(f"[Phase 1.{chunk_idx}] Warning: Failed to read state, consecutive_no_progress={programmatic_no_progress}")
                if programmatic_no_progress >= self.config.convergence_chunks:
                    print(f"[Phase 1.{chunk_idx}] Stopping: Convergence detected - "
                          f"{programmatic_no_progress} chunks without progress")
                    break

            # 检查 exploration_summary.md 是否已生成（Claude 可能提前完成）
            # 只有在覆盖率达标时才因为 summary 存在而停止
            summary_file_path = Path(output_dir) / self.EXPLORATION_SUMMARY_FILE
            if summary_file_path.exists():
                latest_state = self._read_exploration_state(output_dir)
                if latest_state and latest_state.coverage_ratio >= self.config.min_coverage_threshold:
                    print(f"[Phase 1.{chunk_idx}] Exploration summary found and coverage reached {latest_state.coverage_ratio:.1%}")
                    break
                else:
                    # 覆盖率未达标，备份 summary 文件继续探索
                    coverage_pct = latest_state.coverage_ratio if latest_state else 0
                    print(f"[Phase 1.{chunk_idx}] Summary exists but coverage {coverage_pct:.1%} < {self.config.min_coverage_threshold:.1%}, backing up and continuing...")
                    # 备份当前 summary（而不是删除）
                    backup_name = f"exploration_summary_chunk{chunk_idx}.md"
                    summary_file_path.rename(Path(output_dir) / backup_name)
                    print(f"[Phase 1.{chunk_idx}] Backed up summary to {backup_name}")

        # 最终状态检查
        final_state = self._read_exploration_state(output_dir)
        if final_state:
            print(f"\n[Phase 1] Exploration completed:")
            print(f"[Phase 1]   Total chunks: {chunk_idx}")
            print(f"[Phase 1]   Coverage: {final_state.coverage_ratio:.1%}")
            print(f"[Phase 1]   Tested functions: {final_state.tested_functions}")
            print(f"[Phase 1]   Complete: {final_state.exploration_complete}")
            if final_state.completion_reason:
                print(f"[Phase 1]   Reason: {final_state.completion_reason}")

        # Phase 1.5 & 1.6: Goal 生成和 Goal-Driven 执行
        task_instruction = None  # 用于 Phase 3
        if enable_goal_driven:
            # 读取探索总结文件
            exploration_summary = self._read_exploration_summary(output_dir)
            if not exploration_summary:
                print("\n[Phase 1.5] Warning: No exploration summary found, skipping Goal-driven phase")
                enable_goal_driven = False
            else:
                # Phase 1.5: Goal 生成
                print("\n[Phase 1.5] Generating task instruction and execution guide...")
                available_skills = self._extract_skill_names(skills_dir)

                # 使用 LLM 从探索总结生成任务指令和执行指南
                task_instruction, execution_guide = self.exploration_summarizer.generate_goal(
                    exploration_summary=exploration_summary,
                    available_skills=available_skills,
                    file_summaries=file_summaries,
                )
                print("="*80)
                print(f"{exploration_summary}")
                print(f"{available_skills}")
                print(f"{file_summaries}")
                print("="*80)
                print(f"[Phase 1.5] Task instruction: {task_instruction}...")
                print(f"[Phase 1.5] Execution guide: {execution_guide if execution_guide else '[None]'}...")
                print("="*80)

                # Phase 1.6: Goal-Driven 执行
                print("\n[Phase 1.6] Goal-driven execution...")

                # 彻底清理探索阶段创建的所有文件和目录
                self._cleanup_exploration_artifacts(
                    trajectory=trajectory,
                    output_dir=output_dir,
                    input_files=input_files,
                )

                goal_prompt = self.prompt_builder.build_goal_driven_prompt(
                    task_instruction=task_instruction,
                    execution_guide=execution_guide,
                    skills_hint=skills_dir,  # 与探索模板一致
                    file_summaries=file_summaries,  # 与探索模板一致
                    output_dir=output_dir,  # 与探索输出目录一致
                )

                print("============Goal Prompt============")
                print(goal_prompt)
                print("="*80)

                # 执行 Goal-Driven（带重试）
                max_retries = 3
                for attempt in range(max_retries):
                    final_result = self.executor.execute(
                        prompt=goal_prompt,
                        skills_dir=skills_dir,
                        input_file=primary_input_file,
                    )

                    if not final_result.success and not final_result.has_trajectory:
                        print(f"[Phase 1.6] Attempt {attempt + 1} failed: {final_result.error or 'Unknown error'}")
                        if attempt < max_retries - 1:
                            print(f"[Phase 1.6] Retrying...")
                            continue
                        raise SynthesisError(f"Goal-driven execution failed after {max_retries} attempts: {final_result.error or 'Unknown error'}")

                    # **关键**：仅使用 Goal-Driven 的轨迹进行后续处理
                    trajectory = self.recorder.record(final_result)
                    print(f"[Phase 1.6] Final trajectory: {trajectory.num_steps} steps")

                    if trajectory.num_steps > 1:
                        break
                    elif attempt < max_retries - 1:
                        print(f"[Phase 1.6] Attempt {attempt + 1}: Only {trajectory.num_steps} steps. Retrying...")

                result = final_result

                # 保存 Goal-Driven 最终轨迹
                final_trajectory_path = self._save_raw_trajectory(
                    result=final_result,
                    trajectory=trajectory,
                    output_dir=output_dir,
                    task_id=f"{task_id or 'task'}_goal_driven_final",
                )
                print(f"[Phase 1.6] Final trajectory saved: {final_trajectory_path}")

        if not enable_goal_driven:
            # 不使用 Goal-Driven，直接使用最后一次探索的轨迹
            print("\n[Phase 1.5] Goal-driven disabled, using last exploration trajectory")
            raw_trajectory_path = self._save_raw_trajectory(
                result=result,
                trajectory=trajectory,
                output_dir=output_dir,
                task_id=task_id,
            )
            print(f"[Phase 1] Raw trajectory saved to: {raw_trajectory_path}")

        # Phase 2: Trajectory Processing
        print("\n[Phase 2] Processing trajectory...")
        validation = self.processor.validate(trajectory)
        if not validation.is_valid:
            # Show debug info before failing
            print(f"[Phase 2] Validation errors: {validation.errors}")
            print(f"[Phase 2] Debug - Steps recorded:")
            for step in trajectory.steps[:5]:
                print(f"  - {step.action_type}: {step.tool_name or step.reasoning[:50] if step.reasoning else 'N/A'}...")
            if trajectory.raw_events:
                print(f"[Phase 2] Debug - Raw events count: {len(trajectory.raw_events)}")
            raise InvalidTrajectoryError(validation.errors)

        if validation.warnings:
            for warning in validation.warnings:
                print(f"[Phase 2] Warning: {warning}")

        processed = self.processor.process(trajectory)
        print(f"[Phase 2] Used skills: {processed.used_skills}")
        print(f"[Phase 2] Pip packages: {processed.pip_packages}")

        # Phase 3: 指令微调
        print("\n[Phase 3] Instruction refinement...")

        # 读取输出文件内容用于验证
        output_contents = self._read_output_files(trajectory.output_files)

        if task_instruction:
            # 如果有 Phase 1.5 生成的指令，验证一致性
            print("[Phase 3] Verifying instruction consistency...")
            is_consistent, reason = self.instruction_generator.verify_instruction_consistency(
                instruction=task_instruction,
                trajectory=trajectory,
                output_files_content=output_contents,
            )

            if is_consistent:
                print("[Phase 3] Instruction is consistent with trajectory")
                final_instruction = task_instruction
            else:
                print(f"[Phase 3] Instruction inconsistent: {reason}")
                print("[Phase 3] Refining instruction...")
                final_instruction = self.instruction_generator.refine_instruction(
                    original_instruction=task_instruction,
                    trajectory=trajectory,
                    inconsistency_reason=reason,
                    output_files_content=output_contents,
                )
                print(f"[Phase 3] Refined instruction: {final_instruction[:200]}...")
        else:
            # 没有 Phase 1.5 指令，使用传统方法生成
            print("[Phase 3] No task instruction from Phase 1.5, generating from trajectory...")
            final_instruction = self.instruction_generator.generate(
                trajectory=trajectory,
                input_files=input_files if input_files else None,
            )
            print(f"[Phase 3] Generated instruction: {final_instruction[:200]}...")

        print("\n[Phase 3] Generating tests...")
        tests = self.test_generator.generate(trajectory)

        print("\n[Phase 3] Packaging task...")
        task = self.packager.package(
            task_id=task_id,
            trajectory=trajectory,
            instruction=final_instruction,
            tests=tests,
            input_files=input_files if input_files else None,
            used_skills=processed.used_skills,
            pip_packages=processed.pip_packages,
            skills_dir=skills_dir,
            output_dir=output_dir,
        )

        print(f"\n[Pipeline] Task generated: {task.task_path}")
        return task

    def synthesize_from_trajectory(
        self,
        trajectory_data: dict[str, Any],
        input_file: str | None = None,
        skills_dir: str | None = None,
        output_dir: str = "./workspaces",
        task_id: str | None = None,
        save_raw: bool = True,
    ) -> HarborTask:
        """
        Synthesize a task from existing trajectory data.

        Useful when you already have trajectory data from a previous execution.

        Args:
            trajectory_data: Raw trajectory data (events list)
            input_file: Path to input file
            skills_dir: Path to skills directory
            output_dir: Output directory
            task_id: Optional task ID
            save_raw: Whether to save raw trajectory for debugging

        Returns:
            HarborTask with generated task information
        """
        from .execution.claude_executor import ExecutionResult

        # Create mock result
        result = ExecutionResult(
            success=True,
            output="",
            exit_code=0,
            trajectory_data=trajectory_data,
        )

        # Record trajectory
        trajectory = self.recorder.record(result)

        # Save raw trajectory for debugging comparison
        if save_raw:
            raw_path = self._save_raw_trajectory(
                result=result,
                trajectory=trajectory,
                output_dir=output_dir,
                task_id=task_id,
            )
            print(f"[Debug] Raw trajectory saved to: {raw_path}")

        # Validate
        validation = self.processor.validate(trajectory)
        if not validation.is_valid:
            raise InvalidTrajectoryError(validation.errors)

        # Process
        processed = self.processor.process(trajectory)

        # Generate instruction
        instruction = self.instruction_generator.generate(
            trajectory=trajectory,
            input_files=[input_file] if input_file else None,
        )

        # Generate tests
        tests = self.test_generator.generate(trajectory)

        # Package
        return self.packager.package(
            task_id=task_id,
            trajectory=trajectory,
            instruction=instruction,
            tests=tests,
            input_files=[input_file] if input_file else None,
            used_skills=processed.used_skills,
            pip_packages=processed.pip_packages,
            skills_dir=skills_dir,
            output_dir=output_dir,
        )

    def synthesize_batch(
        self,
        configs: list[dict[str, Any]],
        output_dir: str = "./workspaces",
    ) -> list[HarborTask]:
        """
        Synthesize multiple tasks from a list of configurations.

        Args:
            configs: List of config dicts with keys: input_file, skills_dir, domain, etc.
            output_dir: Output directory for all tasks

        Returns:
            List of generated HarborTask objects
        """
        tasks = []

        for i, config in enumerate(configs):
            print(f"\n{'='*60}")
            print(f"Processing task {i+1}/{len(configs)}")
            print(f"{'='*60}")

            try:
                task = self.synthesize(
                    input_file=config.get("input_file"),
                    input_files=config.get("input_files"),
                    skills_dir=config.get("skills_dir"),
                    prompt_style=config.get("prompt_style", "minimal"),
                    max_steps=config.get("max_steps", 10),
                    domain=config.get("domain"),
                    output_dir=output_dir,
                    task_id=config.get("task_id"),
                    skip_summary=config.get("skip_summary", False),
                )
                tasks.append(task)
                print(f"Success: {task.task_path}")
            except Exception as e:
                print(f"Failed: {e}")
                continue

        return tasks


# =========================================================================
# V2: DAG-aware pipeline
# =========================================================================


class HarborSynthesisPipelineV2(HarborSynthesisPipeline):
    """DAG-aware pipeline for synthesizing Harbor tasks from skill dependency graphs."""

    DAG_EXPLORATION_STATE_FILE = "dag_exploration_state.json"

    def run(
        self,
        dag_task_input: str | dict,
        entity_folder: str | None,
        output_dir: str,
        task_id: str | None = None,
        isolated: bool = True,
        workspace_root: str = "./workspaces",
    ) -> HarborTask:
        """Full pipeline: parse DAG -> explore -> synthesize."""
        import uuid

        print("=" * 80)
        print("[HarborSynthesisPipelineV2] Starting DAG-based pipeline")
        print("=" * 80)

        dag_task = self._parse_dag_task(dag_task_input)
        print(f"[V2] DAG Task: {dag_task.task_id} ({dag_task.structure_type})")
        print(f"[V2] Skills: {dag_task.skill_names}")
        print(f"[V2] Paths: {len(dag_task.all_paths())}")

        exploration_summary_path = self.phase1_dag_exploration(
            dag_task=dag_task,
            entity_folder=entity_folder,
            output_dir=output_dir,
        )

        if not task_id:
            task_id = f"dag_task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # Prepare file_summary for phase2
        file_summary: FileSummaryResult | str = FileSummaryResult()
        file_summary_json = Path(output_dir) / self.FILE_SUMMARY_FILE
        if file_summary_json.exists():
            file_summary = str(file_summary_json)
        elif entity_folder and Path(entity_folder).exists():
            file_summary = self.phase1_file_summary(entity_folder)

        skills_dir = self._resolve_skills_dir(dag_task)

        return self.phase2_dag_task_synthesis(
            dag_task=dag_task,
            exploration_summary_path=exploration_summary_path,
            file_summary=file_summary,
            skills_dir=skills_dir,
            output_dir=output_dir,
            task_id=task_id,
            isolated=isolated,
            workspace_root=workspace_root,
        )

    def phase1_dag_exploration(
        self,
        dag_task: DAGTask,
        entity_folder: str | None,
        output_dir: str,
        pre_built_file_summaries: dict[str, str] | None = None,
    ) -> str:
        """
        DAG-guided exploration phase.

        Args:
            dag_task: DAGTask object
            entity_folder: Path to entity folder (optional)
            output_dir: Output directory
            pre_built_file_summaries: Pre-built file summaries (path -> summary).
                If provided, skips LLM-based file summarization.

        Returns:
            Path to exploration_summary.md
        """
        from .utils.file_utils import (
            setup_exploration_workspace,
            cleanup_exploration_workspace,
            copy_exploration_results,
        )

        output_dir = str(Path(output_dir).resolve())
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n[Phase 1 - DAG Exploration]")
        print(f"[Phase 1] Structure: {dag_task.structure_type}")
        print(f"[Phase 1] Paths to cover: {len(dag_task.all_paths())}")
        print(f"[Phase 1] Edges to verify: {len(dag_task.edges)}")

        skills_dir = self._resolve_skills_dir(dag_task)
        print(f"[Phase 1] Skills directory: {skills_dir}")

        file_summaries: dict[str, str] = {}
        if pre_built_file_summaries:
            file_summaries = pre_built_file_summaries
            print(f"[Phase 1] Using pre-built file summaries: {len(file_summaries)} files")
        elif entity_folder and Path(entity_folder).exists():
            file_summary_result = self.phase1_file_summary(entity_folder)
            file_summaries = {
                entry.path: entry.summary for entry in file_summary_result.files
            }
            print(f"[Phase 1] Entity files: {len(file_summaries)}")

        input_files = list(file_summaries.keys())
        workspace_info = setup_exploration_workspace(
            skills_dir=skills_dir,
            input_files=input_files,
            file_summaries=file_summaries,
        )
        workspace_path = workspace_info["workspace"]
        input_file_mapping = workspace_info.get("input_file_mapping", {})
        working_dir = str(workspace_info["output"]) if isinstance(workspace_info["output"], Path) else str(workspace_info["output"])
        print(f"[Phase 1] Workspace: {workspace_path}")

        workspace_file_summaries = {}
        for orig_path, summary in file_summaries.items():
            ws_path = input_file_mapping.get(str(Path(orig_path).resolve()))
            if ws_path:
                workspace_file_summaries[ws_path] = summary
            else:
                workspace_file_summaries[orig_path] = summary

        dag_state = DAGExplorationState.from_dag_task(dag_task)
        self._save_dag_exploration_state(dag_state, working_dir)

        skills_docs = self._read_skill_docs(dag_task)

        max_chunks = self.config.max_dag_exploration_chunks
        prompt_config = PromptConfig(template_style=PromptStyle.MINIMAL, max_steps=20)
        self.prompt_builder = PromptBuilder(prompt_config)

        programmatic_no_progress = 0
        primary_input_file = next(iter(workspace_file_summaries.keys())) if workspace_file_summaries else None

        try:
            for chunk_idx in range(1, max_chunks + 1):
                print(f"\n[Phase 1.{chunk_idx}] Exploration chunk {chunk_idx}/{max_chunks}")

                current_state = self._read_dag_exploration_state(working_dir) or dag_state

                print(f"[Phase 1.{chunk_idx}] Path coverage (state file): {current_state.path_coverage:.1%}, "
                      f"Edge coverage: {current_state.edge_coverage:.1%}")

                if current_state.exploration_complete or current_state.path_coverage >= self.config.dag_path_coverage_threshold:
                    print(f"[Phase 1.{chunk_idx}] Stopping: coverage target reached")
                    break

                if programmatic_no_progress >= self.config.dag_convergence_chunks:
                    print(f"[Phase 1.{chunk_idx}] Stopping: convergence")
                    break

                previous_summary = None
                if chunk_idx > 1:
                    prev_path = Path(working_dir) / f"exploration_summary_chunk{chunk_idx - 1}.md"
                    if prev_path.exists():
                        previous_summary = prev_path.read_text(encoding="utf-8")

                prompt = self.prompt_builder.build_dag_exploration_prompt(
                    dag_task=dag_task,
                    dag_state=current_state,
                    file_summaries=workspace_file_summaries,
                    skills_docs=skills_docs,
                    output_dir=working_dir,
                    chunk_index=chunk_idx,
                    checkpoint_interval=self.config.checkpoint_interval,
                    previous_summary=previous_summary,
                )

                result = self.executor.execute(
                    prompt=prompt,
                    working_dir=working_dir,
                    skills_dir=None,
                    input_file=primary_input_file,
                )

                if not result.success and not result.has_trajectory:
                    print(f"[Phase 1.{chunk_idx}] Execution failed: {result.error}")
                    programmatic_no_progress += 1
                    continue

                trajectory = self.recorder.record(result)
                print(f"[Phase 1.{chunk_idx}] Recorded {trajectory.num_steps} steps")

                updated_state = self._read_dag_exploration_state(working_dir)
                if updated_state:
                    prev_coverage = current_state.path_coverage
                    new_coverage = updated_state.path_coverage
                    if new_coverage > prev_coverage:
                        programmatic_no_progress = 0
                    elif updated_state.exploration_complete:
                        break
                    else:
                        checkpoint_files = list(Path(working_dir).glob("checkpoint_*.md"))
                        has_progress = (
                            len(checkpoint_files) >= chunk_idx
                            or trajectory.num_steps >= 10
                        )
                        if has_progress:
                            programmatic_no_progress = 0
                        else:
                            programmatic_no_progress += 1
                            print(f"[Phase 1.{chunk_idx}] No progress detected ({programmatic_no_progress}/{self.config.dag_convergence_chunks})")

                    if updated_state.path_coverage >= self.config.dag_path_coverage_threshold:
                        break
                else:
                    programmatic_no_progress += 1

                workspace_summary = Path(working_dir) / self.EXPLORATION_SUMMARY_FILE
                if workspace_summary.exists():
                    latest = self._read_dag_exploration_state(working_dir)
                    if latest and latest.path_coverage >= self.config.dag_path_coverage_threshold:
                        break
                    else:
                        backup_name = f"exploration_summary_chunk{chunk_idx}.md"
                        workspace_summary.rename(Path(working_dir) / backup_name)

            print(f"\n[Phase 1] Copying results to: {output_dir}")
            copied_files = copy_exploration_results(working_dir, output_dir)
            print(f"[Phase 1] Copied {len(copied_files)} files")

        finally:
            print(f"\n[Phase 1] Cleaning up workspace")
            cleanup_exploration_workspace(workspace_path)

        summary_path = Path(output_dir) / self.EXPLORATION_SUMMARY_FILE
        if not summary_path.exists():
            print(f"[Phase 1] Generating fallback summary...")
            final_state = self._read_dag_exploration_state(output_dir)
            if final_state:
                fallback = self._generate_fallback_dag_summary(output_dir, dag_task, final_state)
                summary_path.write_text(fallback, encoding="utf-8")
            else:
                summary_path.write_text(
                    f"# DAG Exploration Summary\n\nSkills: {dag_task.skill_names}\nStructure: {dag_task.structure_type}\n",
                    encoding="utf-8",
                )

        print(f"[Phase 1] Exploration complete: {summary_path}")
        return str(summary_path)

    def phase2_dag_task_synthesis(
        self,
        dag_task: DAGTask,
        exploration_summary_path: str,
        file_summary: FileSummaryResult | str,
        skills_dir: str,
        output_dir: str,
        target_files: list[str] | None = None,
        task_id: str | None = None,
        isolated: bool = True,
        workspace_root: str = "./workspaces",
    ) -> HarborTask:
        """
        DAG-constrained task synthesis (full pipeline).

        This is a complete task synthesis method that mirrors phase3_task_synthesis
        but uses DAG-constrained instruction generation (generate_task_from_dag)
        instead of the generic generate_task_from_exploration.

        子步骤：
        2.1 过滤文件（如果指定了 target_files）
        2.2 合成任务指令 — DAG-constrained (dag_task_synthesize.py)
        2.2.5 质量过滤
        2.3 合成引导思路 (guiding_metadata.py)
        2.4 生成 oracle 轨迹
        2.4.5 PRM 轨迹验证
        2.4.6 提取轨迹摘要
        2.4.7 生成 pytest 测试
        2.4.8 solve.sh 生成与验证
        2.4.9 打包（skillsbench 格式）

        Args:
            dag_task: DAGTask object with DAG structure info
            exploration_summary_path: 探索报告路径
            file_summary: FileSummaryResult 对象或 JSON 文件路径
            skills_dir: 技能目录路径
            output_dir: 输出目录路径
            target_files: 用户指定的目标文件列表（None 则使用全部）
            task_id: 任务 ID（可选）
            isolated: 是否使用隔离工作空间（默认 True）
            workspace_root: 隔离工作空间根目录

        Returns:
            HarborTask 对象
        """
        import uuid
        from .utils.file_utils import setup_isolated_workspace

        print(f"\n[Phase 2] DAG Task Synthesis")

        # 生成任务 ID
        if not task_id:
            task_id = f"dag_task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # 读取探索报告
        exploration_summary = Path(exploration_summary_path).read_text(encoding="utf-8")
        print(f"[Phase 2] Loaded exploration summary: {len(exploration_summary)} chars")

        # 加载 file_summary（记录原始路径）
        file_summary_path: str | None = None
        if isinstance(file_summary, str):
            file_summary_path = file_summary
            file_summary = self.file_summarizer.load_from_json(file_summary)

        # 2.1 过滤文件
        if target_files:
            print(f"[Phase 2.1] Filtering to {len(target_files)} target files")
            file_summary = file_summary.filter_by_paths(target_files)
        else:
            print(f"[Phase 2.1] Using all {len(file_summary.files)} files")

        if not file_summary.files:
            raise SynthesisError("No files available for task synthesis")

        # ========== 创建隔离工作空间 ==========
        working_dir: str | None = None
        skills_dir_effective = skills_dir
        output_dir_effective = output_dir
        harbor_output_dir = output_dir

        if isolated:
            original_input_files = [entry.path for entry in file_summary.files]
            workspace = setup_isolated_workspace(
                task_id=task_id,
                skills_dir=skills_dir,
                input_files=original_input_files,
                exploration_summary_path=exploration_summary_path,
                file_metadata_path=file_summary_path,
                workspace_root=workspace_root,
            )

            print(f"[Phase 2] Created isolated workspace: {workspace['workspace']}")

            # 更新路径为隔离环境中的路径
            working_dir = str(workspace["workspace"])
            skills_path = workspace["skills"]
            if isinstance(skills_path, Path):
                skills_dir_effective = str(skills_path)  # .claude/skills/ contains skill subdirs directly
            output_path = workspace["output"]
            if isinstance(output_path, Path):
                output_dir_effective = str(output_path)
            harbor_path = workspace["harbor_task"]
            if isinstance(harbor_path, Path):
                harbor_output_dir = str(harbor_path)

            # 更新 file_summary 中的文件路径为隔离环境中的路径
            copied_files = workspace.get("input_files", [])
            if isinstance(copied_files, list):
                path_mapping = {
                    Path(original).name: str(copied)
                    for original, copied in zip(original_input_files, copied_files)
                }
                for entry in file_summary.files:
                    new_path = path_mapping.get(Path(entry.path).name)
                    if new_path:
                        entry.path = new_path
        # ==========================================

        # 提取可用技能
        available_skills = self._extract_skill_names(skills_dir)

        # 2.2 合成任务指令 — DAG-constrained
        print(f"[Phase 2.2] Generating DAG-constrained task instruction...")
        task_instruction = self.instruction_generator.generate_task_from_dag(
            dag_task=dag_task,
            exploration_summary=exploration_summary,
            file_summaries=file_summary,
        )
        print(f"[Phase 2.2] Task instruction:")
        print("=" * 60)
        print(task_instruction)
        print("=" * 60)

        # 2.2.5 质量过滤 - 评分并可能拒绝低质量指令（含 DAG 合规性检查）
        print(f"[Phase 2.2.5] Filtering task instruction quality (with DAG compliance)...")
        try:
            filter_result = self.instruction_generator.filter_task_instruction(
                task_instruction=task_instruction,
                max_retries=3,
                dag_task=dag_task,
            )

            weighted_avg = filter_result.get("weighted_average", {}).get("score", 0)
            normalized_score = weighted_avg / 10.0  # Convert 0-10 to 0-1

            print(f"[Phase 2.2.5] Instruction quality score: {weighted_avg}/10 ({normalized_score:.2f})")

            # Log individual dimension scores
            for dim in ["goal_clarity", "input_clarity", "constraint_completeness",
                        "referential_clarity", "verifiability_uniqueness", "human_likeness"]:
                if dim in filter_result:
                    score = filter_result[dim].get("score", "N/A")
                    reason = filter_result[dim].get("reason", "")[:80]
                    print(f"  - {dim}: {score}/10 ({reason})")

            if normalized_score < 0.8:
                raise SynthesisError(
                    f"Task instruction quality too low: {weighted_avg}/10 (threshold: 8.0/10). "
                    f"Consider regenerating with different exploration context."
                )

            # DAG compliance check
            dag_compliance = filter_result.get("dag_compliance")
            if dag_compliance:
                dag_avg = dag_compliance.get("dag_weighted_average", {}).get("score", 0)
                dag_normalized = dag_avg / 10.0
                print(f"[Phase 2.2.5] DAG compliance score: {dag_avg}/10 ({dag_normalized:.2f})")
                for dim in ["skill_coverage", "topological_consistency", "edge_semantics"]:
                    if dim in dag_compliance:
                        score = dag_compliance[dim].get("score", "N/A")
                        reason = dag_compliance[dim].get("reason", "")[:80]
                        print(f"  - {dim}: {score}/10 ({reason})")

                if dag_normalized < 0.7:
                    raise SynthesisError(
                        f"DAG compliance too low: {dag_avg}/10 (threshold: 7.0/10). "
                        f"Instruction does not sufficiently respect DAG structure."
                    )

            print(f"[Phase 2.2.5] Instruction passed quality filter")

        except ValueError as e:
            raise SynthesisError(f"Instruction quality filter failed: {e}")

        # 2.3 合成引导思路（含 DAG workflow 约束）
        print(f"[Phase 2.3] Generating guiding metadata (with DAG workflow)...")
        guiding_metadata = self.instruction_generator.generate_guiding_metadata(
            task_instruction=task_instruction,
            exploration_summary=exploration_summary,
            file_summaries=file_summary,
            skills=available_skills,
            dag_task=dag_task,
        )
        print(f"[Phase 2.3] Guiding metadata:")
        print("=" * 60)
        print(guiding_metadata)
        print("=" * 60)

        # 2.4 生成 oracle 轨迹
        print(f"[Phase 2.4] Generating oracle trajectory...")

        # 准备文件摘要
        file_summaries_dict = {entry.path: entry.summary for entry in file_summary.files}

        # 构建执行 prompt（使用任务指令 + 引导思路）
        execution_prompt = self.prompt_builder.build_goal_driven_prompt(
            task_instruction=task_instruction,
            execution_guide=guiding_metadata,
            skills_hint=skills_dir_effective,
            file_summaries=file_summaries_dict,
            output_dir=output_dir_effective,
        )

        # 使用第一个文件作为主输入
        primary_input_file = file_summary.files[0].path if file_summary.files else None

        # 执行任务
        # 注意：当使用隔离工作空间时，技能已经被复制到 .claude/skills/ 下，
        # 不需要再传 skills_dir 给 executor
        result = self.executor.execute(
            prompt=execution_prompt,
            working_dir=working_dir,
            skills_dir=None if isolated else skills_dir_effective,
            input_file=primary_input_file,
        )

        if not result.success and not result.has_trajectory:
            raise SynthesisError(f"Oracle trajectory generation failed: {result.error}")

        trajectory = self.recorder.record(result)
        print(f"[Phase 2.4] Oracle trajectory: {trajectory.num_steps} steps")

        # ========== Phase 2.4.5: PRM 轨迹验证 ==========
        if self.config.enable_prm_validation:
            print(f"[Phase 2.4.5] PRM trajectory validation...")
            prm_valid = False

            for prm_attempt in range(self.config.max_prm_retries):
                print(f"[Phase 2.4.5] PRM validation attempt {prm_attempt + 1}/{self.config.max_prm_retries}...")

                prm_result = self.trajectory_validator.validate(
                    trajectory=trajectory,
                    task_instruction=task_instruction,
                    skills_dir=skills_dir_effective,
                    working_dir=working_dir,
                )

                if prm_result.is_valid:
                    print(f"[Phase 2.4.5] PRM validation passed!")
                    prm_valid = True
                    break

                print(f"[Phase 2.4.5] PRM validation failed: {prm_result.issues}")
                print(f"[Phase 2.4.5] Feedback: {prm_result.feedback[:500]}...")

                if prm_attempt < self.config.max_prm_retries - 1:
                    # Retry oracle with PRM feedback
                    retry_prompt = self.trajectory_validator.build_retry_prompt(
                        original_prompt=execution_prompt,
                        validation_result=prm_result,
                        attempt_number=prm_attempt + 2,
                    )

                    result = self.executor.execute(
                        prompt=retry_prompt,
                        working_dir=working_dir,
                        skills_dir=None if isolated else skills_dir_effective,
                        input_file=primary_input_file,
                    )

                    if not result.success and not result.has_trajectory:
                        print(f"[Phase 2.4.5] Oracle retry {prm_attempt + 2} failed: {result.error}")
                        continue

                    trajectory = self.recorder.record(result)
                    print(f"[Phase 2.4.5] Oracle retry {prm_attempt + 2} trajectory: {trajectory.num_steps} steps")

            if not prm_valid:
                raise SynthesisError(
                    f"PRM validation failed after {self.config.max_prm_retries} attempts. "
                    f"Pipeline terminated. Issues: {prm_result.issues}"
                )
        else:
            print(f"[Phase 2.4.5] PRM validation disabled, skipping...")

        # ========== Phase 2.4.6: 提取轨迹摘要并保存最终结果 ==========
        print(f"[Phase 2.4.6] Extracting trajectory summary and saving final results...")

        # 提取轨迹摘要（截断版本，用于后续生成）
        trajectory_summary = self.trajectory_validator.extract_trajectory_for_prm(
            trajectory=trajectory,
            max_response_chars=self.config.prm_max_response_chars,
        )
        print(f"[Phase 2.4.6] Trajectory summary extracted: {len(trajectory_summary)} chars")

        # 保存最终结果到 final_res 目录
        # 清理可能存在的残留文件，防止前一个任务的输出被错误地用于测试生成
        final_res_dir = Path(working_dir) / "final_res" if working_dir else Path("/tmp/final_res")
        if final_res_dir.exists():
            import shutil as shutil_cleanup
            shutil_cleanup.rmtree(final_res_dir)
            print(f"[Phase 2.4.6] Cleaned up existing final_res directory")
        final_res_dir.mkdir(parents=True, exist_ok=True)

        # 保存轨迹摘要
        trajectory_summary_path = final_res_dir / "trajectory_summary.txt"
        trajectory_summary_path.write_text(trajectory_summary, encoding="utf-8")

        # 复制输出文件到 final_res
        final_output_files = []
        for output_file in trajectory.output_files:
            src = Path(output_file)
            if src.exists():
                import shutil
                dst = final_res_dir / src.name
                shutil.copy2(src, dst)
                final_output_files.append(str(dst))
                print(f"[Phase 2.4.6] Copied output: {src.name}")

        print(f"[Phase 2.4.6] Final results saved to: {final_res_dir}")

        # ========== Phase 2.4.7: 生成 pytest 测试（skillsbench 风格） ==========
        print(f"[Phase 2.4.7] Generating pytest tests (skillsbench style)...")

        pytest_test_path = final_res_dir / "test_outputs.py"
        pytest_content = None
        pytest_passed = False
        pytest_test_result = None

        for pytest_attempt in range(self.config.max_pytest_retries):
            print(f"[Phase 2.4.7] Pytest generation attempt {pytest_attempt + 1}/{self.config.max_pytest_retries}...")

            try:
                if pytest_attempt == 0:
                    # 首次生成
                    pytest_content = self.pytest_generator.generate(
                        task_instruction=task_instruction,
                        trajectory_summary=trajectory_summary,
                        final_files=final_output_files,
                        working_dir=str(final_res_dir),
                        test_file_path=str(pytest_test_path),
                    )
                else:
                    # 根据失败信息重新生成
                    failure_info = pytest_test_result.get_failure_summary() if pytest_test_result else "Unknown error"
                    pytest_content = self.pytest_generator.regenerate_with_feedback(
                        task_instruction=task_instruction,
                        trajectory_summary=trajectory_summary,
                        final_files=final_output_files,
                        previous_test=pytest_content,
                        failure_info=failure_info,
                        working_dir=str(final_res_dir),
                        test_file_path=str(pytest_test_path),
                    )

                # 保存测试文件
                pytest_test_path.write_text(pytest_content, encoding="utf-8")

                # 验证语法
                is_valid_syntax, syntax_error = self.pytest_generator.validate_test_syntax(pytest_content)
                if not is_valid_syntax:
                    print(f"[Phase 2.4.7] Syntax error in generated tests: {syntax_error}")
                    continue

            except Exception as e:
                print(f"[Phase 2.4.7] Test generation failed: {e}")
                continue

            # 在 final_res 目录运行 pytest 验证
            print(f"[Phase 2.4.7] Running pytest against final results...")
            pytest_test_result = self.test_executor.run_tests(
                test_file=str(pytest_test_path),
                working_dir=str(final_res_dir),
                timeout=self.config.test_timeout,
            )

            if pytest_test_result.all_passed:
                print(f"[Phase 2.4.7] Pytest passed! ({pytest_test_result.passed}/{pytest_test_result.total})")
                pytest_passed = True
                break

            print(f"[Phase 2.4.7] Pytest failed: {pytest_test_result.get_failure_summary()}")

        if not pytest_passed:
            failure_info = pytest_test_result.get_failure_summary() if pytest_test_result else "Test generation failed"
            raise SynthesisError(
                f"Pytest generation failed after {self.config.max_pytest_retries} attempts. "
                f"Pipeline terminated. Failures: {failure_info}"
            )

        # ========== Phase 2.4.6-2.4.7 (Legacy): 计算代码验证 ==========
        # 保留旧逻辑作为备用，当 pytest 生成成功时跳过
        if self.config.enable_computation_tests and not pytest_passed:
            print(f"[Phase 2.4.6] Generating computation-based tests...")

            computation_test_path = Path(working_dir) / "computation_tests.py" if working_dir else Path("/tmp/computation_tests.py")
            computation_passed = False
            comp_test_result = None

            for comp_attempt in range(self.config.max_computation_retries):
                print(f"[Phase 2.4.6] Computation test generation attempt {comp_attempt + 1}/{self.config.max_computation_retries}...")

                try:
                    if comp_attempt == 0 or comp_test_result is None:
                        # First attempt or previous attempt failed before test execution: generate new tests
                        computation_tests = self.computation_test_generator.generate(
                            trajectory=trajectory,
                            task_instruction=task_instruction,
                            input_files=[entry.path for entry in file_summary.files],
                            output_files=trajectory.output_files,
                            working_dir=working_dir,
                            skills_dir=skills_dir_effective,
                            test_file_path=str(computation_test_path),
                        )
                    else:
                        # Subsequent attempts: regenerate with failure feedback
                        computation_tests = self.computation_test_generator.regenerate(
                            trajectory=trajectory,
                            task_instruction=task_instruction,
                            failure_summary=comp_test_result.get_failure_summary(),
                            working_dir=working_dir,
                            skills_dir=skills_dir_effective,
                            test_file_path=str(computation_test_path),
                        )

                    # Save computation tests
                    computation_test_path.write_text(computation_tests, encoding="utf-8")

                    # Validate syntax
                    is_valid_syntax, syntax_error = self.computation_test_generator.validate_test_syntax(computation_tests)
                    if not is_valid_syntax:
                        print(f"[Phase 2.4.6] Syntax error in generated tests: {syntax_error}")
                        continue

                except Exception as e:
                    print(f"[Phase 2.4.6] Test generation failed: {e}")
                    continue

                # Run computation tests
                print(f"[Phase 2.4.7] Running computation tests...")
                comp_test_result = self.test_executor.run_tests(
                    test_file=str(computation_test_path),
                    working_dir=working_dir,
                    timeout=self.config.test_timeout,
                )

                # Debug output
                print(f"[Phase 2.4.7] DEBUG: passed={comp_test_result.passed}, failed={comp_test_result.failed}, "
                      f"errors={comp_test_result.errors}, total={comp_test_result.total}, "
                      f"return_code={comp_test_result.return_code}, all_passed={comp_test_result.all_passed}")

                if comp_test_result.all_passed:
                    print(f"[Phase 2.4.7] Computation tests passed! ({comp_test_result.passed}/{comp_test_result.total})")
                    computation_passed = True
                    break

                print(f"[Phase 2.4.7] Computation tests failed: {comp_test_result.get_failure_summary()}")

            if not computation_passed:
                failure_info = comp_test_result.get_failure_summary() if comp_test_result else "Test generation failed"
                raise SynthesisError(
                    f"Computation verification failed after {self.config.max_computation_retries} attempts. "
                    f"Pipeline terminated. Failures: {failure_info}"
                )
        else:
            print(f"[Phase 2.4.6-2.4.7] Computation tests disabled, skipping...")

        # 验证轨迹
        validation = self.processor.validate(trajectory)
        if not validation.is_valid:
            raise InvalidTrajectoryError(validation.errors)

        processed = self.processor.process(trajectory)

        # 准备 input_files（在打包中需要）
        input_files = [entry.path for entry in file_summary.files]

        # ========== Phase 2.4.8: solve.sh 生成与验证（干净工作空间） ==========
        if self.config.enable_solve_verification:
            print(f"[Phase 2.4.8] Generating and validating solve.sh...")

            # solve.sh 路径
            solve_sh_path = Path(working_dir) / "solve.sh" if working_dir else Path("/tmp/solve.sh")

            # 生成初始 solve.sh（使用 trajectory_summary）
            if self.config.use_claude_for_solve and working_dir:
                print(f"[Phase 2.4.8] Using Claude Code to generate solve.sh...")
                solve_sh_content = self.solve_generator.generate_with_trajectory_summary(
                    task_instruction=task_instruction,
                    trajectory_summary=trajectory_summary,
                    input_files=input_files,
                    executor=self.executor,
                    working_dir=working_dir,
                    solve_sh_path=str(solve_sh_path),
                )
            else:
                # 使用 LLM 生成（fallback）
                solve_sh_content = self.solve_generator.generate_solve_sh(
                    trajectory=trajectory,
                    task_instruction=task_instruction,
                    expectation_tests_content=pytest_content,
                )

            # 在干净工作空间验证 solve.sh（使用 Phase 2.4.7 生成的 pytest）
            solve_passed = False
            solve_test_result = None
            verify_workspace = None

            for solve_attempt in range(self.config.max_solve_retries + 1):
                if solve_attempt > 0:
                    # 重试：根据失败信息重新生成 solve.sh
                    if self.config.use_claude_for_solve and working_dir:
                        print(f"[Phase 2.4.8] Retry {solve_attempt}/{self.config.max_solve_retries}: Refining solve.sh with Claude Code...")
                        solve_sh_content = self.solve_generator.refine_with_executor(
                            previous_solve_sh=solve_sh_content,
                            test_failures=solve_test_result.failures,
                            executor=self.executor,
                            working_dir=working_dir,
                            solve_sh_path=str(solve_sh_path),
                            task_instruction=task_instruction,
                            trajectory_summary=trajectory_summary,
                        )
                    else:
                        print(f"[Phase 2.4.8] Retry {solve_attempt}/{self.config.max_solve_retries}: Regenerating solve.sh with LLM...")
                        solve_sh_content = self.solve_generator.regenerate_with_llm(
                            trajectory=trajectory,
                            task_instruction=task_instruction,
                            test_failures=solve_test_result.failures,
                            previous_solve_sh=solve_sh_content,
                            expectation_tests_content=pytest_content,
                        )

                # 在干净工作空间验证
                print(f"[Phase 2.4.8] Verifying solve.sh in clean workspace...")
                solve_passed, solve_test_result, verify_workspace = self.solve_verifier.verify_in_clean_workspace(
                    solve_sh_content=solve_sh_content,
                    test_file_content=pytest_content,
                    input_files=input_files,
                    skills_dir=skills_dir_effective,
                    timeout=self.config.solve_timeout,
                    conda_env=self.config.conda_env,
                    cleanup=False,  # 保留工作空间用于调试
                )

                if solve_passed:
                    print(f"[Phase 2.4.8] solve.sh validation passed!")
                    break

                print(f"[Phase 2.4.8] solve.sh validation failed: {solve_test_result.get_failure_summary()}")

            if not solve_passed:
                print(f"[Phase 2.4.8] ERROR: solve.sh still failing after {self.config.max_solve_retries} retries")
                raise SolveSynthesisError(
                    f"solve.sh validation failed after {self.config.max_solve_retries} retries: "
                    f"{solve_test_result.get_failure_summary()}"
                )
        else:
            # 禁用验证时，仍生成 solve.sh 但不验证
            solve_sh_content = self.solve_generator.generate_from_trajectory(trajectory)
        # ================================================

        # ========== Phase 2.4.9: 打包（skillsbench 格式） ==========
        print(f"[Phase 2.4.9] Packaging task in skillsbench format...")

        # 路径规范化
        skillsbench_normalizer = SkillsbenchPathNormalizer(
            workspace_path=working_dir,
            input_files=input_files,
        )
        # 规范化所有文件的路径：instruction、pytest、solve.sh
        instruction_normalized = skillsbench_normalizer.normalize(task_instruction)
        pytest_content_normalized = skillsbench_normalizer.normalize_tests(pytest_content)
        solve_sh_normalized = skillsbench_normalizer.normalize_solve_sh(solve_sh_content)

        # 使用 skillsbench 格式打包
        task = self.packager.package_skillsbench_format(
            task_id=task_id,
            instruction=instruction_normalized,
            tests_content=pytest_content_normalized,
            solve_sh_content=solve_sh_normalized,
            input_files=input_files,
            skills_dir=skills_dir_effective,
            output_dir=harbor_output_dir,
            metadata={
                "difficulty": self.config.default_difficulty,
                "category": self.config.default_category,
            },
            pip_packages=processed.pip_packages,
            used_skills=processed.used_skills,
            output_files=trajectory.output_files,
        )

        # 保存引导思路到任务目录
        guiding_path = Path(task.task_path) / "guiding_metadata.md"
        guiding_path.write_text(guiding_metadata, encoding="utf-8")
        print(f"[Phase 2] Guiding metadata saved to: {guiding_path}")

        # 移动中间产物文件到单独目录，保持 harbor 任务目录干净
        if isolated and working_dir:
            self._move_synthesis_artifacts(
                task_path=task.task_path,
                workspace_dir=working_dir,
            )

        print(f"\n[Phase 2] Task generated: {task.task_path}")
        if isolated:
            print(f"[Phase 2] Workspace: {working_dir}")
        return task

    # =========================================================================
    # DAG-specific helpers
    # =========================================================================

    def _parse_dag_task(self, dag_task_input: str | dict) -> DAGTask:
        """Parse DAG task from file path or dict."""
        if isinstance(dag_task_input, str):
            path = Path(dag_task_input)
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    data = data[0]
            else:
                data = json.loads(dag_task_input)
        else:
            data = dag_task_input
        return DAGTask.from_json(data)

    def _resolve_skills_dir(self, dag_task: DAGTask) -> str:
        """Resolve the skills directory from DAG task skill_doc_paths."""
        if dag_task.skill_doc_paths:
            first_path = next(iter(dag_task.skill_doc_paths.values()))
            skills_parent = Path(first_path).parent.parent
            if skills_parent.exists():
                return str(skills_parent)
        return ""

    def _read_skill_docs(self, dag_task: DAGTask) -> dict[str, str]:
        """Read SKILL.md content for each skill in the DAG."""
        docs: dict[str, str] = {}
        for skill_name in dag_task.skill_names:
            doc_path = dag_task.skill_doc_paths.get(skill_name, "")
            if doc_path and Path(doc_path).exists():
                try:
                    docs[skill_name] = Path(doc_path).read_text(encoding="utf-8")
                except Exception as e:
                    docs[skill_name] = f"[Error reading {doc_path}: {e}]"
            else:
                docs[skill_name] = f"[No documentation available for {skill_name}]"
        return docs

    def _save_dag_exploration_state(self, state: DAGExplorationState, output_dir: str) -> None:
        """Save DAG exploration state to file."""
        state_path = Path(output_dir) / self.DAG_EXPLORATION_STATE_FILE
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps(state.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _read_dag_exploration_state(self, output_dir: str) -> DAGExplorationState | None:
        """Read DAG exploration state from file."""
        state_path = Path(output_dir) / self.DAG_EXPLORATION_STATE_FILE
        if not state_path.exists():
            return None
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            return DAGExplorationState.from_dict(data)
        except Exception as e:
            print(f"[V2] Warning: Failed to read DAG state: {e}")
            return None

    def _generate_fallback_dag_summary(
        self,
        output_dir: str,
        dag_task: DAGTask,
        state: DAGExplorationState,
    ) -> str:
        """Generate a fallback exploration summary using LLM."""
        from .prompts.dag_exploration import FALLBACK_DAG_SUMMARY_PROMPT

        dag_structure = (
            f"Structure: {dag_task.structure_type}\n"
            f"Skills: {dag_task.skill_names}\n"
            f"Edges: {[str(e) for e in dag_task.edges]}\n"
            f"Topological order: {dag_task.topological_order()}"
        )

        checkpoint_content = self._read_checkpoint_files(output_dir)

        prompt = FALLBACK_DAG_SUMMARY_PROMPT.format(
            dag_structure=dag_structure,
            exploration_state_json=json.dumps(state.to_dict(), indent=2, ensure_ascii=False),
            checkpoint_files_content=checkpoint_content,
        )

        summary = self.llm_client_synthesis.generate(
            system_prompt="You are an expert at summarizing DAG-based skill exploration results.",
            user_prompt=prompt,
            temperature=0.3,
        )
        return summary


def main():
    """CLI entry point for skillnet_gym.synthesis"""
    parser = argparse.ArgumentParser(
        description="Harbor Task Auto-Synthesis Pipeline V2 (DAG-aware)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # DAG-based pipeline (new V2 mode)
  python -m skillnet_gym.synthesis --dag-task /path/to/dag_task.json \\
      --entity-folder /path/to/entities --output /path/to/output

  # Legacy mode (same as v1 phases)
  python -m skillnet_gym.synthesis --phase all \\
      --entity-folder /path/to/pdfs --skills-dir skills/pdf --output tasks/

  # Phase 2: Skill Exploration (new exploration)
  python -m skillnet_gym.synthesis --phase exploration \\
      --file-summary file_summaries.json --skills-dir skills/pdf --output exploration/

  # Phase 2: Skill Exploration (reuse existing)
  python -m skillnet_gym.synthesis --phase exploration \\
      --reuse-exploration /root/output/xxx/exploration_summary.md

  # Phase 3: Task Synthesis (all files)
  python -m skillnet_gym.synthesis --phase task_synthesis \\
      --exploration exploration_summary.md --file-summary file_summaries.json \\
      --skills-dir skills/pdf --output tasks/

  # Phase 3: Task Synthesis (specific target files)
  python -m skillnet_gym.synthesis --phase task_synthesis \\
      --exploration exploration_summary.md --file-summary file_summaries.json \\
      --skills-dir skills/pdf --target-files /path/to/fw4.pdf /path/to/sc100.pdf --output tasks/

  # Phase 3: Batch Task Synthesis (one task per file, concurrent)
  python -m skillnet_gym.synthesis --phase task_synthesis --batch \\
      --exploration exploration_summary.md --file-summary file_summaries.json \\
      --skills-dir skills/pdf --max-workers 3 --output tasks/

  # Full pipeline (all phases)
  python -m skillnet_gym.synthesis --phase all \\
      --entity-folder /path/to/pdfs --skills-dir skills/pdf --output tasks/

  # Full pipeline with exploration reuse
  python -m skillnet_gym.synthesis --phase all \\
      --entity-folder /path/to/pdfs --skills-dir skills/pdf \\
      --reuse-exploration /root/output/xxx/exploration_summary.md --output tasks/

  # Legacy mode (backward compatible)
  python -m skillnet_gym.synthesis --input data.xlsx --skills-dir skills/
        """,
    )

    # V2 DAG-specific argument
    parser.add_argument(
        "--dag-task",
        type=str,
        help="Path to DAG task JSON file (activates V2 DAG pipeline mode)",
    )

    # Phase selection argument
    parser.add_argument(
        "--phase",
        type=str,
        choices=["file_summary", "exploration", "task_synthesis", "all"],
        default=None,
        help="Pipeline phase to execute (file_summary, exploration, task_synthesis, or all)",
    )

    # Phase 1: File Summary arguments
    parser.add_argument(
        "--entity-folder",
        type=str,
        help="Path to entity folder containing input files (for file_summary phase)",
    )
    parser.add_argument(
        "--file-extensions",
        type=str,
        nargs="+",
        help="File extensions to process (e.g., .pdf .xlsx)",
    )
    parser.add_argument(
        "--ignore-files",
        type=str,
        nargs="+",
        help="File names to ignore (e.g., requirements.txt download_report.json)",
    )
    parser.add_argument(
        "--extract-metadata",
        action="store_true",
        default=False,
        help="Extract detailed metadata for each file (slower but richer, for task_synthesis)",
    )

    # Phase 2: Exploration arguments
    parser.add_argument(
        "--file-summary",
        type=str,
        help="Path to file_summaries.json (for exploration/task_synthesis phases)",
    )
    parser.add_argument(
        "--reuse-exploration",
        type=str,
        help="Path to existing exploration_summary.md to reuse",
    )

    # Phase 3: Task Synthesis arguments
    parser.add_argument(
        "--exploration",
        type=str,
        help="Path to exploration_summary.md (for task_synthesis phase)",
    )
    parser.add_argument(
        "--target-files",
        type=str,
        nargs="+",
        help="Specific target files for task synthesis (optional, uses all if not specified)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: generate one task per file concurrently (for task_synthesis phase)",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Use all files from file_summary instead of selecting representative files (for exploration phase)",
    )

    # Isolation settings for task synthesis
    parser.add_argument(
        "--no-isolated",
        action="store_true",
        help="Disable isolated workspace (use original file paths directly)",
    )
    parser.add_argument(
        "--workspace-root",
        type=str,
        default="./workspaces",
        help="Root directory for isolated workspaces (default: ./workspaces)",
    )

    # Legacy arguments (backward compatibility)
    parser.add_argument(
        "--input", "-i",
        type=str,
        action="append",
        dest="input_files",
        help="[Legacy] Path to input file (can be specified multiple times)",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="[Legacy] Skip Phase 0 file summary generation",
    )
    parser.add_argument(
        "--skills-dir", "-s",
        type=str,
        help="Path to skills directory (required for exploration/task_synthesis phases)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory (required for exploration phase, default: ./workspaces for task_synthesis)",
    )
    parser.add_argument(
        "--prompt-style",
        type=str,
        choices=["minimal", "domain_guided", "skill_hinted", "goal_oriented"],
        default="minimal",
        help="Prompt template style (default: minimal)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum exploration steps (default: 10)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        help="Domain hint for the task (e.g., 'data_analysis', 'document_processing')",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        help="Custom task ID (auto-generated if not specified)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-opus-4-5-20251124",
        help="Claude model to use for execution",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o",
        help="LLM model for instruction synthesis (default: gpt-4o)",
    )
    parser.add_argument(
        "--llm-model-synthesis",
        type=str,
        default=None,
        help="Override: LLM model for synthesis tasks (default: same as --llm-model)",
    )
    parser.add_argument(
        "--llm-model-verification",
        type=str,
        default="gpt-4o",
        help="LLM model for verification tasks (PRM, pytest, etc. default: gpt-4o)",
    )
    parser.add_argument(
        "--llm-api-key",
        type=str,
        help="API key for LLM (defaults to LLM_API_KEY env var)",
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default="https://api.openai.com/v1",
        help="Base URL for LLM API (default: https://api.openai.com/v1)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=30,
        help="Maximum exploration chunks (safety limit, default: 30)",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.9,
        help="Minimum coverage threshold to stop exploration (default: 0.9)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=20,
        help="Number of tool calls between checkpoints (default: 20)",
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default="base",
        help="Conda environment for Claude Code execution (default: base)",
    )
    parser.add_argument(
        "--no-goal-driven",
        action="store_true",
        help="Disable Goal-driven execution",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Maximum concurrent workers for file summary and task synthesis (default: 3)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar display",
    )
    parser.add_argument(
        "--no-require-output",
        action="store_true",
        help="Skip validation that requires output files (useful for debugging)",
    )

    args = parser.parse_args()

    # Create config - only override defaults if args are provided
    config_kwargs = {
        "model": args.model,
        "llm_model": args.llm_model,
        "llm_model_synthesis": args.llm_model_synthesis or args.llm_model,
        "llm_model_verification": args.llm_model_verification,
        "llm_base_url": args.llm_base_url,
        "max_exploration_chunks": args.max_chunks,
        "min_coverage_threshold": args.min_coverage,
        "checkpoint_interval": args.checkpoint_interval,
        "conda_env": args.conda_env,
        "max_workers": args.max_workers,
        "show_progress": not args.no_progress,
        "require_output_files": not args.no_require_output,
    }
    # Only set these if provided (to preserve defaults from PipelineConfig)
    if args.llm_api_key:
        config_kwargs["llm_api_key"] = args.llm_api_key
    if args.output:
        config_kwargs["output_dir"] = args.output

    config = PipelineConfig(**config_kwargs)

    try:
        # Determine execution mode
        dag_task_arg = getattr(args, "dag_task", None)
        if dag_task_arg:
            # V2 DAG pipeline mode
            pipeline_v2 = HarborSynthesisPipelineV2(config)
            output_dir = args.output or "./workspaces"
            isolated = not args.no_isolated
            task = pipeline_v2.run(
                dag_task_input=dag_task_arg,
                entity_folder=getattr(args, "entity_folder", None),
                output_dir=output_dir,
                task_id=getattr(args, "task_id", None),
                isolated=isolated,
                workspace_root=args.workspace_root,
            )
            print("\n" + "=" * 60)
            print("DAG PIPELINE COMPLETE")
            print("=" * 60)
            print(f"Task ID: {task.task_id}")
            print(f"Task Path: {task.task_path}")
        elif args.phase:
            # Legacy phase-based execution (v1 compatible)
            pipeline = HarborSynthesisPipeline(config)
            _run_phase_based(pipeline, args)
        elif args.input_files:
            # Legacy execution mode
            pipeline = HarborSynthesisPipeline(config)
            _run_legacy_mode(pipeline, args)
        else:
            print("Error: Either --dag-task, --phase, or --input is required", file=sys.stderr)
            parser.print_help()
            sys.exit(1)

    except InvalidTrajectoryError as e:
        print(f"\nError: Invalid trajectory - {e.errors}", file=sys.stderr)
        sys.exit(1)
    except SynthesisError as e:
        print(f"\nError: Synthesis failed - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _run_phase_based(pipeline: HarborSynthesisPipeline, args: argparse.Namespace):
    """Run pipeline in phase-based mode."""
    phase = args.phase

    if phase == "file_summary":
        # Phase 1: File Summary
        if not args.entity_folder:
            print("Error: --entity-folder is required for file_summary phase", file=sys.stderr)
            sys.exit(1)

        result = pipeline.phase1_file_summary(
            entity_folder=args.entity_folder,
            output_json=args.output,
            file_extensions=args.file_extensions,
            ignore_files=args.ignore_files,
            extract_metadata=args.extract_metadata,
        )

        print("\n" + "=" * 60)
        print("FILE SUMMARY COMPLETE")
        print("=" * 60)
        print(f"Total files: {len(result.files)}")
        print(f"Content types: {list(result.content_types.keys())}")
        print(f"Output: {args.output}")

    elif phase == "exploration":
        # Phase 2: Skill Exploration
        if args.reuse_exploration:
            # Just verify the exploration file exists
            if not Path(args.reuse_exploration).exists():
                print(f"Error: Exploration file not found: {args.reuse_exploration}", file=sys.stderr)
                sys.exit(1)
            print(f"\nReusing existing exploration: {args.reuse_exploration}")
            return

        if not args.file_summary:
            print("Error: --file-summary is required for exploration phase", file=sys.stderr)
            sys.exit(1)
        if not args.skills_dir:
            print("Error: --skills-dir is required for exploration phase", file=sys.stderr)
            sys.exit(1)
        if not args.output:
            print("Error: --output is required for exploration phase", file=sys.stderr)
            sys.exit(1)

        summary_path = pipeline.phase2_exploration(
            file_summary=args.file_summary,
            skills_dir=args.skills_dir,
            output_dir=args.output,
            existing_exploration=args.reuse_exploration,
            use_all_files=args.all_files,
        )

        print("\n" + "=" * 60)
        print("EXPLORATION COMPLETE")
        print("=" * 60)
        print(f"Exploration summary: {summary_path}")

    elif phase == "task_synthesis":
        # Phase 3: Task Synthesis
        exploration_path = args.exploration or args.reuse_exploration
        if not exploration_path:
            print("Error: --exploration is required for task_synthesis phase", file=sys.stderr)
            sys.exit(1)
        if not args.file_summary:
            print("Error: --file-summary is required for task_synthesis phase", file=sys.stderr)
            sys.exit(1)
        if not args.skills_dir:
            print("Error: --skills-dir is required for task_synthesis phase", file=sys.stderr)
            sys.exit(1)

        # Use default output dir if not specified
        task_output_dir = args.output or "./workspaces"

        # Isolation settings
        isolated = not args.no_isolated
        workspace_root = args.workspace_root

        if args.batch:
            # Batch mode: generate one task per file concurrently
            target_file_groups = None
            if args.target_files:
                # User specified files: generate one task per specified file
                target_file_groups = [[f] for f in args.target_files]

            tasks = pipeline.phase3_batch_task_synthesis(
                exploration_summary_path=exploration_path,
                file_summary=args.file_summary,
                skills_dir=args.skills_dir,
                output_dir=task_output_dir,
                target_file_groups=target_file_groups,
                isolated=isolated,
                workspace_root=workspace_root,
            )

            print("\n" + "=" * 60)
            print("BATCH TASK SYNTHESIS COMPLETE")
            print("=" * 60)
            print(f"Generated {len(tasks)} tasks")
            for task in tasks:
                print(f"  - {task.task_id}: {task.task_path}")

        else:
            # Single task mode
            task = pipeline.phase3_task_synthesis(
                exploration_summary_path=exploration_path,
                file_summary=args.file_summary,
                skills_dir=args.skills_dir,
                output_dir=task_output_dir,
                target_files=args.target_files,
                task_id=args.task_id,
                isolated=isolated,
                workspace_root=workspace_root,
            )

            print("\n" + "=" * 60)
            print("TASK SYNTHESIS COMPLETE")
            print("=" * 60)
            print(f"Task ID: {task.task_id}")
            print(f"Task Path: {task.task_path}")
            print(f"Input Files: {task.input_files}")
            print(f"Output Files: {task.output_files}")
        print(f"Used Skills: {task.used_skills}")
        print("\nTo validate the task:")
        print(f"  harbor tasks check {task.task_path}")
        print("\nTo run with oracle:")
        print(f"  harbor run -p {task.task_path} -a oracle")

    elif phase == "all":
        # Full pipeline (all three phases)
        if not args.entity_folder:
            print("Error: --entity-folder is required for full pipeline", file=sys.stderr)
            sys.exit(1)
        if not args.skills_dir:
            print("Error: --skills-dir is required for full pipeline", file=sys.stderr)
            sys.exit(1)

        # Isolation settings
        isolated = not args.no_isolated
        workspace_root = args.workspace_root

        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1
        file_summary_path = output_dir / pipeline.FILE_SUMMARY_FILE
        file_summary = pipeline.phase1_file_summary(
            entity_folder=args.entity_folder,
            output_json=str(file_summary_path),
            file_extensions=args.file_extensions,
            ignore_files=args.ignore_files,
        )

        # Phase 2
        exploration_path = pipeline.phase2_exploration(
            file_summary=file_summary,
            skills_dir=args.skills_dir,
            output_dir=str(output_dir),
            existing_exploration=args.reuse_exploration,
            use_all_files=args.all_files,
        )

        # Phase 3
        task = pipeline.phase3_task_synthesis(
            exploration_summary_path=exploration_path,
            file_summary=file_summary,
            skills_dir=args.skills_dir,
            output_dir=str(output_dir),
            target_files=args.target_files,
            task_id=args.task_id,
            isolated=isolated,
            workspace_root=workspace_root,
        )

        print("\n" + "=" * 60)
        print("FULL PIPELINE COMPLETE")
        print("=" * 60)
        print(f"File Summary: {file_summary_path}")
        print(f"Exploration: {exploration_path}")
        print(f"Task ID: {task.task_id}")
        print(f"Task Path: {task.task_path}")
        print(f"Input Files: {task.input_files}")
        print(f"Output Files: {task.output_files}")
        print(f"Used Skills: {task.used_skills}")
        print("\nTo validate the task:")
        print(f"  harbor tasks check {task.task_path}")
        print("\nTo run with oracle:")
        print(f"  harbor run -p {task.task_path} -a oracle")


def _run_legacy_mode(pipeline: HarborSynthesisPipeline, args: argparse.Namespace):
    """Run pipeline in legacy mode (backward compatible with old CLI)."""
    if not args.skills_dir:
        print("Error: --skills-dir is required for legacy mode", file=sys.stderr)
        sys.exit(1)

    task = pipeline.synthesize(
        input_files=args.input_files,
        skills_dir=args.skills_dir,
        prompt_style=args.prompt_style,
        max_steps=args.max_steps,
        domain=args.domain,
        output_dir=args.output,
        task_id=args.task_id,
        skip_summary=args.skip_summary,
        no_goal_driven=args.no_goal_driven,
    )

    print("\n" + "=" * 60)
    print("SYNTHESIS COMPLETE")
    print("=" * 60)
    print(f"Task ID: {task.task_id}")
    print(f"Task Path: {task.task_path}")
    print(f"Input Files: {task.input_files}")
    print(f"Output Files: {task.output_files}")
    print(f"Used Skills: {task.used_skills}")
    print("\nTo validate the task:")
    print(f"  harbor tasks check {task.task_path}")
    print("\nTo run with oracle:")
    print(f"  harbor run -p {task.task_path} -a oracle")


if __name__ == "__main__":
    main()
