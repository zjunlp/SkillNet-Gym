# Harbor Synthesis V2 Architecture

## Overview

`skillnet_gym.synthesis` is a DAG-aware task auto-synthesis pipeline that generates verifiable coding tasks (Harbor Tasks) by orchestrating Claude Code executions. It extends v1 with support for directed acyclic graph (DAG) structured skill dependencies, enabling multi-skill task generation that respects topological ordering and edge constraints.

The pipeline takes input files + skill definitions, explores the skills autonomously, then synthesizes a complete task package including: instruction, solve.sh, pytest tests, and a Dockerfile — all validated end-to-end before output.

## Module Structure

```
skillnet_gym.synthesis/
├── __init__.py              # Public API exports
├── __main__.py              # CLI entry point
├── config.py                # All data structures and configuration
├── pipeline.py              # Main pipeline orchestration (HarborSynthesisPipeline, V2)
├── execution/               # Claude Code execution layer
│   ├── claude_executor.py   # Subprocess wrapper for Claude CLI
│   ├── prompt_builder.py    # Prompt template construction
│   └── trajectory_recorder.py  # JSONL event → structured Trajectory
├── synthesis/               # Task generation & validation components
│   ├── instruction_generator.py   # LLM-based task instruction synthesis
│   ├── exploration_summarizer.py  # Exploration loop summarization
│   ├── file_summarizer.py         # File content summarization
│   ├── metadata_extractor.py      # Guiding metadata extraction
│   ├── solve_generator.py         # solve.sh generation from trajectory
│   ├── solve_verifier.py          # Clean-workspace solve.sh validation
│   ├── trajectory_processor.py    # Trajectory processing & validation
│   ├── trajectory_validator.py    # PRM (Process Reward Model) validation
│   ├── computation_test_generator.py  # Computation-based test generation
│   ├── pytest_generator.py        # Pytest test generation (skillsbench style)
│   ├── test_executor.py           # Test runner (pytest subprocess)
│   ├── test_generator.py          # Legacy test generation
│   ├── task_packager.py           # Final task directory assembly
│   └── path_normalizer.py         # Path normalization for portability
├── prompts/                 # Prompt templates (system/user prompts)
│   ├── dag_exploration.py         # DAG exploration prompts
│   ├── dag_task_synthesize.py     # DAG-constrained instruction synthesis
│   ├── task_synthesize.py         # Generic task instruction synthesis
│   ├── guiding_metadata.py        # Execution guide generation
│   ├── filter.py                  # Quality scoring prompts
│   ├── prm_validation_prompts.py  # PRM validation prompts
│   ├── pytest_prompts.py          # Pytest generation prompts
│   ├── solve_sh_prompts.py        # solve.sh generation prompts
│   ├── computation_test_prompts.py  # Computation test prompts
│   ├── expectation_test_prompts.py  # Expectation test prompts
│   └── representative_file_selection.py  # File selection prompts
├── utils/                   # Shared utilities
│   ├── llm_client.py       # OpenAI-compatible LLM client wrapper
│   ├── file_utils.py       # File/directory operations, workspace management
│   ├── path_normalizer.py  # Skillsbench path normalization
│   └── dependency_extractor.py  # pip/apt dependency extraction
└── env_builder/             # Conda environment construction
    ├── analyze_skill_deps.py    # Skill dependency analysis
    ├── merge_environments.py    # Multi-skill environment merging
    └── build_envs.sh           # Shell script for building conda envs
```

## Core Data Flow

```
Input Files + Skills Directory + (optional) DAG Task JSON
                          │
                          ▼
              ┌─────────────────────┐
              │  Phase 1: File      │
              │  Summarization      │
              │  (LLM per file)     │
              └─────────┬───────────┘
                        │ FileSummaryResult
                        ▼
              ┌─────────────────────┐
              │  Phase 2: Skill     │
              │  Exploration        │
              │  (Claude Code ×N)   │
              │  Checkpoint-driven  │
              └─────────┬───────────┘
                        │ exploration_summary.md
                        ▼
              ┌─────────────────────┐
              │  Phase 3: Task      │
              │  Synthesis          │
              │                     │
              │  3.2 Instruction    │
              │  3.2.5 Quality Gate │
              │  3.3 Guide Metadata │
              │  3.4 Oracle Traj    │
              │  3.4.5 PRM Valid    │
              │  3.4.7 Pytest Gen   │
              │  3.4.8 solve.sh     │
              │  3.4.9 Package      │
              └─────────┬───────────┘
                        │
                        ▼
              Harbor Task Package
              (instruction.md, solve.sh, tests/,
               input/, skills/, Dockerfile, task.toml)
```

## Pipeline Classes

### `HarborSynthesisPipeline` (Base)

The base pipeline supporting single/multi-skill task synthesis with three-phase workflow:

1. **Phase 1 (File Summary)**: Uses Claude Code CLI to summarize each input file, producing a `FileSummaryResult` with content type classification (form/text/table/code/mixed).

2. **Phase 2 (Exploration)**: Runs Claude Code in an isolated workspace with checkpoint-based adaptive exploration. Convergence is detected via:
   - Coverage threshold (default 90% of documented functions tested)
   - Convergence detection (N consecutive chunks without progress)
   - Safety limit (max chunk count)
   
   Each chunk is a full Claude Code session (~20 tool calls). State is persisted to `exploration_state.json` between chunks.

3. **Phase 3 (Task Synthesis)**: Multi-step generation with validation loops:
   - 3.2: LLM generates a task instruction from exploration findings
   - 3.2.5: Quality filter scores the instruction (threshold: 8.0/10)
   - 3.3: LLM generates execution guide (guiding metadata)
   - 3.4: Claude Code executes the task to produce an oracle trajectory
   - 3.4.5: PRM validates the trajectory (retry loop)
   - 3.4.7: LLM generates pytest tests; validates against oracle output
   - 3.4.8: Generates solve.sh; validates in clean workspace
   - 3.4.9: Packages everything into skillsbench format

### `HarborSynthesisPipelineV2` (DAG-Aware)

Extends the base pipeline with DAG-constrained task synthesis:

- Accepts a `DAGTask` JSON defining skill nodes, directed edges, and suggested workflows
- Exploration follows topological order, ensuring source skills are explored before target skills
- Path coverage (all source-to-sink paths) must reach 100%
- Instruction generation respects DAG edge constraints (data handoffs between skills)
- Supports DAG structures: chain, fan_out, fan_in, diamond

## Key Components

### Execution Layer

**`ClaudeExecutor`**: Wraps the Claude CLI (`~/.local/bin/claude`) as a subprocess. Features:
- Isolated working directory with `.claude_runtime/` config
- Conda environment support (injects env into PATH)
- Long prompt handling via temp file piping
- Trajectory extraction from JSONL session files
- Configurable model, max_turns, timeout

**`PromptBuilder`**: Constructs prompts for different phases:
- `CHECKPOINT_EXPLORATION_PROMPT`: Multi-skill exploration with state tracking
- `GOAL_DRIVEN_TEMPLATE`: Task execution with instruction + guide
- `DAG_EXPLORATION_PROMPT`: DAG-guided exploration
- Supports styles: minimal, domain_guided, skill_hinted, goal_oriented

**`TrajectoryRecorder`**: Parses Claude Code JSONL events into a structured `Trajectory`:
- Extracts tool uses, text blocks, and tool results
- Tracks files read/written, skills used, bash commands executed
- Truncates tool outputs to prevent context overflow

### Synthesis Layer

**`InstructionGenerator`**: Uses LLM (synthesis model) to generate task instructions from exploration context. Includes quality filtering with multi-dimensional scoring.

**`TrajectoryValidator` (PRM)**: Validates oracle trajectories by calling an LLM to verify correctness. Supports retry with feedback.

**`PytestGenerator`**: Generates pytest tests that validate task outputs. Tests are run against oracle results to confirm they pass before packaging.

**`SolveShGenerator`**: Converts oracle trajectory into a deterministic bash script. Can use either Claude Code or LLM for generation. Validated in a clean workspace.

**`TaskPackager`**: Assembles the final task directory in skillsbench format:
```
task_xxx/
├── instruction.md
├── solve.sh
├── tests/test_outputs.py
├── input/           # Input files
├── skills/          # Skill definitions (copied to multiple agent paths)
├── Dockerfile
└── task.toml        # Metadata (difficulty, category, timeouts)
```

### LLM Configuration

The pipeline uses two LLM roles with separate model configurations:
- **Synthesis** (`llm_model_synthesis`): Instruction generation, filtering, metadata — uses `gemini-3.1-pro-preview`
- **Verification** (`llm_model_verification`): PRM validation, pytest generation, computation tests — uses `gpt-5.4-pro`

Both share the same API key and base URL (OpenAI-compatible endpoint).

Claude Code execution uses a separate model (`claude-opus-4-5-20251124`) via the Claude CLI.

## DAG Data Structures

**`DAGTask`**: Defines a multi-skill task as a directed graph:
- `skills`: List of skill nodes with IDs and names
- `edges`: Directed edges with alignment types and scenario descriptions
- `structure_type`: chain | fan_out | fan_in | diamond
- Methods: `topological_order()`, `all_paths()`, `source_nodes`, `sink_nodes`

**`DAGExplorationState`**: Tracks DAG exploration progress:
- Path coverage (tested paths / all source-to-sink paths)
- Edge coverage (tested edges / all edges)
- Per-skill core function coverage

## Workspace Isolation

All Claude Code executions run in isolated workspaces:
- Input files are copied to `workspace/input/`
- Skills are copied to `workspace/.claude/skills/`
- Output goes to `workspace/output/`
- Results are copied back to the target directory after completion
- Workspaces are cleaned up after use

For task synthesis, `setup_isolated_workspace()` creates a separate workspace per task to prevent cross-contamination between concurrent task generations.

## Configuration

`PipelineConfig` centralizes all tunable parameters:
- Model settings (model name, max_turns, timeout)
- Exploration settings (max_chunks, coverage_threshold, convergence_chunks)
- Verification settings (PRM retries, pytest retries, solve retries)
- Concurrency settings (max_workers, show_progress)
- Path normalization and trajectory recording limits

## CLI Usage

```bash
# V2 DAG mode
python -m skillnet_gym.synthesis --dag-task dag_task.json \
    --entity-folder /path/to/files --output /path/to/output

# Phase-by-phase execution
python -m skillnet_gym.synthesis --phase file_summary --entity-folder /path/to/files
python -m skillnet_gym.synthesis --phase exploration --file-summary summaries.json --skills-dir skills/
python -m skillnet_gym.synthesis --phase task_synthesis --exploration summary.md --file-summary summaries.json

# Full pipeline (all phases)
python -m skillnet_gym.synthesis --phase all --entity-folder /path/to/files --skills-dir skills/
```
