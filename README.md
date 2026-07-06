<div align="center">
  <h1>SkillNet-Gym: A Dynamic Benchmark for Compositional Skill Learning</h1>
</div>


<p align="center">
  <a href="https://github.com/zjunlp/SkillNet-Gym" target="_blank">📄arXiv</a> •
  <a href="https://github.com/zjunlp/SkillNet-Gym" target="_blank">🤗HFPaper</a> •
  <a href="https://github.com/zjunlp/SkillNet-Gym" target="_blank">📊Dataset</a> •
</p>


<p align="center">
  <a href="https://github.com/zjunlp/SciAtlas">
    <img src="https://awesome.re/badge.svg" alt="Awesome">
  </a>
  <a href="https://github.com/zjunlp/SciAtlas/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
  </a>
  <img src="https://img.shields.io/github/last-commit/zjunlp/SciAtlas?color=blue" alt="Last Commit">
  <img src="https://img.shields.io/badge/PRs-Welcome-red" alt="PRs Welcome">
</p>
<div align="center">
  
  **SkillNet-Gym turns evolving real-world skill ecosystems into executable, quality-controlled benchmark tasks.**
  
</div>

## Table of Contents

- [✨ Overview](#overview)
- [🔧 Installation](#installation)
- [🧭 Benchmark Settings](#benchmark-settings)
- [🗂️ Data Format](#data-format)
- [📊 Evaluation](#evaluation)
- [🛠️ Build Your Own Gym](#build-your-own-skillnet-gym)
- [🎁 Acknowledgement](#acknowledgement)
- [🚩 Citation](#citation)

---

## ✨ Overview


LLM agents increasingly solve complex tasks by using **skills**: reusable procedural assets that may contain instructions, workflow recipes, scripts, templates, examples, or references. However, real skill ecosystems are not fixed. New skills appear, old skills become stale, and useful capabilities often emerge only when multiple skills are retrieved and composed in the right order.

**SkillNet-Gym** is a dynamic benchmark for evaluating compositional skill learning. Instead of freezing a manually curated snapshot, SkillNet-Gym starts from community skills and related web artifacts, organizes them into a continuously extensible **SkillNet**, and automatically synthesizes executable tasks that require agents to construct, retrieve, compose, and apply skills.

<p align="center">
  <img src="imgs/intro.png" alt="" width="92%">
</p>


Concretely, SkillNet-Gym enables:
- **Dynamic task generation from a living SkillNet.** SkillNet-Gym continuously organizes community skills, documents, and processable files into a heterogeneous SkillNet, then samples compositional subgraphs to synthesize new benchmark tasks. As the external skill ecosystem changes, the benchmark can evolve with it instead of becoming a stale snapshot.
- **Unified evaluation of Skill Construction and Skill Composition.** SkillNet-Gym places no-skill solving, official skill usage, model-constructed skills, wild skill retrieval, and orchestration under one benchmark protocol. This allows researchers to locate whether failure comes from poor skill distillation, incomplete retrieval, wrong dependency ordering, incorrect handoffs, or weak final execution.
- **Task-driven self-adaptation of skills.** Each task is backed by files, gold skills, gold dependency edges, reference solutions, and deterministic verifiers. These tasks can serve as optimization targets for self-adaptive agents: failures suggest whether a skill should be rewritten, expanded, split, merged, re-indexed, re-routed, or re-composed with other skills. In this sense, SkillNet-Gym provides an evaluation foundation for an adaptive closed loop: **evaluate → diagnose → adapt skills → execute tasks**.

---

## 🔧 Installation

For end-to-end evaluation, our task format is compatible with [Harbor](https://github.com/harbor-framework/harbor)'s official automated evaluation framework.

```bash
uv tool install harbor
```

To better support users who may have difficulty pulling or running Docker images, we also modified the [Harbor](https://github.com/sunnychenxiwang/harbor/tree/main) source code to **enable execution in a local Conda environment**. In addition, we ensure that Claude Code and Codex agents can run concurrently in isolated workspaces, preventing interference between parallel agent runs.

To support local Harbor evaluation, we provide scripts for the following steps:

```bash
# Installing Claude Code
npm install -g @anthropic-ai/claude-code
# Installing Harbor in a dedicated environment
git clone https://github.com/sunnychenxiwang/harbor.git
conda run -n conda_env pip install -e harbor
# Installing the Conda environments required by the tasks
```

---

## 🧭 Benchmark Metadata

### Benchmark Settings

SkillNet-Gym enable unified evaluation for compositional skill learning.

| Setting | What the Agent Receives | What It Tests | Main Metric |
| --- | --- | --- | --- |
| **No Skill** | Task instruction and files only | Whether the agent can solve the task without procedural support | `Avg@k` pass rate |
| **Skill Efficacy** | Gold official skills attached to the task | Whether provided skills improve execution | `Avg@k` pass rate |
| **Skill Construction** | Upstream documents or community materials | Whether the agent can distill reusable skills before execution | `Avg@k` pass rate after constructed skill use |
| **Skill Retrieve** | A large wild skill pool | Whether the agent can find all gold skills | Completeness / Recall / Precision |
| **Skill Orchestration** | A large wild skill pool | Whether the agent can recover dependency-aware skill workflows | Graph Completeness / Edge Recall / Edge Precision |
| **In the Wild** | A large skill library during task execution | End-to-end performance under realistic repository noise | `Avg@k` pass rate |


---


### Task Information

<p align="center">
  <img src="imgs/taxonomy.png" alt="taxonomy" width="45%" style="margin-right: 20px;">
  <img src="imgs/case_study.png" alt="case study" width="45%">
</p>

---
## 📊 Evaluation

### End-to-end task execution


```text
Avg@k = average pass rate over k independent runs per task
```

Run a full evaluation:

```bash
python -m skillnet_gym.evaluate \
  --setting official_skill \
  --agent codex \
  --model gpt \
  --split all \
  --num-runs 3 \
  --parallel 8 \
  --output runs/official_skill_codex.jsonl
```

Summarize results:

```bash
python -m skillnet_gym.summarize \
  --input runs/official_skill_codex.jsonl \
  --group-by difficulty,domain,topology \
  --output runs/official_skill_codex_summary.md
```

### Skill composition evaluation

Skill composition is evaluated without requiring the agent to execute the full task. The evaluator checks whether the agent retrieves the required skills and reconstructs the correct dependency graph.

```text
Skill Completeness = 1 if the predicted skill set covers all gold skills, else 0
Graph Completeness = 1 if predicted skills and dependency edges cover the gold graph, else 0
Recall / Precision = overlap-based retrieval and edge metrics
Avg selected = average number of selected skills or edges
```

Run composition evaluation:

```bash
python -m skillnet_gym.evaluate_composition \
  --skill-pool data/skillnet/skills.jsonl \
  --tasks data/benchmark/metadata.jsonl \
  --mode retrieve \
  --top-k 10 \
  --output runs/skill_retrieve.jsonl

python -m skillnet_gym.evaluate_composition \
  --skill-pool data/skillnet/skills.jsonl \
  --tasks data/benchmark/metadata.jsonl \
  --mode orchestrate \
  --top-k 10 \
  --output runs/skill_orchestration.jsonl
```

---

## 🛠️ Build Your Own SkillNet-Gym


SkillNet-Gym is not only a fixed benchmark. It is also a recipe for constructing new dynamic skill benchmarks as the skill ecosystem changes.
1. Build Graph
2. Sample
3. Synthesis Tasks


**Building a directed skill graph — and synthesizing verifiable multi-skill coding tasks from it.**

SkillNet-Gym is the open reference implementation for the pipeline described in
*"SkillNet-Gym: Skill-Graph-Driven Auto-Synthesis of Verifiable Multi-Skill Coding
Tasks"*. It contains two complementary sub-pipelines:

1. **Graph construction** (`skillnet_gym.graph`) — search, filter, dedup, and
   scenario-align a corpus of skills into a directed acyclic **skill graph**,
   then sample multi-skill task topologies (chain / fan-in / fan-out / diamond)
   from it.
2. **Task auto-synthesis** (`skillnet_gym.synthesis`) — take a sampled DAG task
   and the skills it references, drive Claude Code through autonomous
   exploration and execution, and package the result as a fully verifiable
   task (`instruction.md`, `solve.sh`, pytest tests, Dockerfile, `task.toml`).

```
   ┌────────────────────────────┐     ┌───────────────────────────────┐
   │  Stage A: Skill Graph      │     │  Stage B: Task Auto-Synthesis │
   │                            │     │                               │
   │  search → filter → dedup   │──▶  │  file summary                 │
   │        ↓                   │     │        ↓                      │
   │  scenario align → edges    │──▶  │  DAG-guided exploration       │
   │        ↓                   │     │        ↓                      │
   │  DAG build → task sample   │──▶  │  instruction / oracle / tests │
   │        ↓                   │     │        ↓                      │
   │  package env + entities    │──▶  │  ➡  Harbor Task package       │
   └────────────────────────────┘     └───────────────────────────────┘
```


   
All credentials come from environment variables (or a `.env` file — the graph
scripts auto-load `.env` from the working directory):

| Variable                 | Used by                       | Default                       |
| ------------------------ | ----------------------------- | ----------------------------- |
| `LLM_API_KEY` / `API_KEY`| every LLM step (both stages)  | —                             |
| `LLM_BASE_URL`           | LLM client                    | `https://api.openai.com/v1`   |
| `LLM_MODEL`              | LLM client                    | `gpt-4o`                      |
| `EMBEDDING_API_KEY`      | dedup + alignment             | falls back to `LLM_API_KEY`   |
| `EMBEDDING_BASE_URL`     | dedup + alignment             | falls back to `LLM_BASE_URL`  |
| `EMBEDDING_MODEL`        | dedup + alignment             | `text-embedding-3-large`      |
| `ANTHROPIC_AUTH_TOKEN`   | Claude Code (synthesis)       | —                             |
| `ANTHROPIC_BASE_URL`     | Claude Code (synthesis)       | `https://api.anthropic.com`   |
| `GITHUB_TOKEN`           | skill download (rate limits)  | —                             |

Any OpenAI-compatible endpoint works (vLLM, OpenRouter, Ollama, etc.).

---

## Stage A — Skill graph construction

The graph pipeline is 14 short, resumable CLI scripts. They all read/write JSON
so you can inspect and re-run any single step. Run in order:

```bash
# 1. Semantic search over SkillNet for candidate skills per query seed
python -m skillnet_gym.graph.search.skillnet_semantic_search \
    --input query_seeds.json --output skillnet_semantic_results.json \
    --limit 30 --threshold 0.8 --workers 8

# 2. Rank / filter by GitHub stars
python -m skillnet_gym.graph.search.filter_skillnet_results \
    --input skillnet_semantic_results.json \
    --output skillnet_semantic_results_by_stars.json \
    --keep 20 --min-stars 10

# 3. Clone the surviving skill repos
python -m skillnet_gym.graph.download.download_filtered_skills \
    --input skillnet_semantic_results_by_stars.json \
    --target-dir downloaded_skills \
    --manifest downloaded_skills_manifest.json \
    --skip-existing --workers 8

# 4. LLM-scored quality gate (cost / verifiability / documentation)
python -m skillnet_gym.graph.download.evaluate_skills_quality \
    --skills-dir downloaded_skills --workers 8

# 5. Embedding cluster + dedup skills
python -m skillnet_gym.graph.dedup.cluster_dedup_downloaded_skills \
    --manifest downloaded_skills_manifest.json \
    --threshold 0.90 --top-neighbors 50

# 6. Extract pre/post scenarios from each SKILL.md
python -m skillnet_gym.graph.scenarios.extract_skill_scenarios --workers 2

# 7. Dedup scenarios via embedding + Louvain clustering
python -m skillnet_gym.graph.dedup.deduplicate_scenarios \
    --top-neighbors 100 --graph-threshold 0.82 --cluster-threshold 0.88

# 8. Match post-scenarios ↔ pre-scenarios and LLM-verify handoffs
python -m skillnet_gym.graph.scenarios.align_skill_scenarios \
    --top-k 30 --min-retrieval-score 0.5 --workers 8

# 9. Review edges for functional redundancy
python -m skillnet_gym.graph.scenarios.review_skill_edge_redundancy \
    --input scenario_alignment_keep.json --workers 8

# 10. Assemble the directed skill graph
python -m skillnet_gym.graph.build_sample.build_scenario_skill_graph \
    --alignments scenario_alignment_nonredundant_keep.json \
    --output scenario_skill_graph.json

# 11. Sample chain / fan-in / fan-out / diamond DAG tasks
python -m skillnet_gym.graph.build_sample.sample_skill_graph_tasks \
    --max-per-category 1000 --output skill_graph_task_candidates.json

# 12. LLM-review composed tasks for compositional validity
python -m skillnet_gym.graph.build_sample.evaluate_skill_graph_tasks \
    --input skill_graph_task_candidates.json --workers 4

# 13. LLM-score candidate input entities against each task
python -m skillnet_gym.graph.packaging.evaluate_task_input_entities \
    --tasks skill_graph_tasks_part_01.json \
    --entities entity/task_input_entities_part_01.json --workers 4

# 14. Materialize per-task environments (copy skills, download inputs)
python -m skillnet_gym.graph.packaging.package_task_environments \
    --tasks 'skill_graph_tasks_*.json' \
    --entities 'entity/task_input_entities_*.json' \
    --output-dir packaged_tasks --workers 8
```

Every step is checkpoint-friendly — most support `--skip-existing`, `--force`,
and `--workers N`. A one-shot driver is in `scripts/run_graph_pipeline.sh`.

---

## Stage B — Task auto-synthesis

Given one packaged task from Stage A (a directory holding a `dag_task.json`,
an `environment/skills/` folder, and input files), synthesize a fully
verifiable coding task:

```bash
# Full DAG-aware pipeline: file summary → exploration → task synthesis
python -m skillnet_gym.synthesis \
    --dag-task packaged_tasks/task-abc123/dag_task.json \
    --entity-folder packaged_tasks/task-abc123/environment \
    --skills-dir packaged_tasks/task-abc123/environment/skills \
    --output ./workspaces
```

Or run phase by phase (useful when iterating on prompts):

```bash
python -m skillnet_gym.synthesis --phase file_summary --entity-folder path/to/files
python -m skillnet_gym.synthesis --phase exploration  --file-summary summaries.json --skills-dir skills/
python -m skillnet_gym.synthesis --phase task_synthesis --exploration summary.md --file-summary summaries.json
```

Output for each task:

```
task_xxx/
├── instruction.md          # LLM-synthesized, quality-filtered
├── solve.sh                # Deterministic oracle solution
├── tests/test_outputs.py   # pytest suite validated against the oracle
├── input/                  # Task input files
├── skills/                 # Skill definitions (SKILL.md + code)
├── Dockerfile              # Reproducible container spec
└── task.toml               # Metadata (difficulty, category, timeouts)
```

### Pipeline internals

The synthesis pipeline runs three phases (see [`docs/architecture.md`](docs/architecture.md)
for details):

| Phase                | Component                                                      | Output                     |
| -------------------- | -------------------------------------------------------------- | -------------------------- |
| **1. File summary**  | `components.file_summarizer`                                    | per-file content type + summary |
| **2. Exploration**   | Claude Code × N checkpointed chunks, DAG-topological           | `exploration_summary.md`   |
| **3. Task synthesis**| instruction → filter → guide → oracle → PRM → pytest → solve.sh | packaged task directory    |

Two LLM roles are used with independent model configuration
(`llm_model_synthesis` / `llm_model_verification`) — both hit the same
OpenAI-compatible endpoint. Claude Code exploration uses a separate Anthropic
model configured via `ANTHROPIC_*`.

---

## Repository layout

```
skillnet-gym/
├── README.md
├── LICENSE                  # Apache 2.0
├── pyproject.toml
├── .env.example
├── docs/
│   └── architecture.md      # Deep-dive on the synthesis pipeline
├── examples/                # Example inputs (query seeds, dag_task.json, …)
├── scripts/                 # One-shot driver scripts
└── src/skillnet_gym/
    ├── graph/
    │   ├── search/          # 1–2  SkillNet search + star filter
    │   ├── download/        # 3–4  GitHub download + LLM quality gate
    │   ├── dedup/           # 5, 7 Skill & scenario embedding clustering
    │   ├── scenarios/       # 6, 8–9 Extraction, alignment, redundancy review
    │   ├── build_sample/    # 10–12 DAG build, topology sampling, task eval
    │   └── packaging/       # 13–14 Entity eval, environment packaging
    └── synthesis/
        ├── pipeline.py      # HarborSynthesisPipeline / …V2 (DAG-aware)
        ├── config.py        # PipelineConfig + all data structures
        ├── execution/       # Claude CLI subprocess wrapper, trajectory recorder
        ├── components/      # Instruction / test / solve.sh generators
        ├── prompts/         # System + user prompt templates
        ├── utils/           # LLM client, file utils, path normalization
        └── env_builder/     # Optional: build conda envs for a task set
```


## 🙏 Acknowledgement
We deeply appreciate the invaluable effort contributed by our dedicated team of developers, supportive users, and esteemed industry partners: [Ant Digital Technologies, Ant Group](https://intl.antdigital.com/en).
This repository develops a benchmark based on [Harbor](https://github.com/harbor-framework/harbor) task types. We sincerely thank all contributors for their outstanding work!



## 🚩 Citation

If SkillNet-Gym is useful in your research, please cite the paper (BibTeX
entry to appear here).
