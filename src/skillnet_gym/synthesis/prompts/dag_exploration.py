"""DAG-guided exploration prompt templates for V2 pipeline."""

DAG_EXPLORATION_SYSTEM = (
    "You are a DAG Workflow Explorer. "
    "You systematically explore skill capabilities by following a directed acyclic graph (DAG) "
    "of skill dependencies. Your exploration must cover every path in the DAG end-to-end."
)

DAG_EXPLORATION_PROMPT = """# DAG-Guided Skill Exploration

You are exploring a set of skills organized as a **Directed Acyclic Graph (DAG)**.
Each node is a skill; each directed edge represents a data handoff from one skill to another.
Your goal is to test every path through the DAG end-to-end.

---

## Skill Dependency Graph

**Structure Type**: {structure_type}

### Nodes (skills)
{nodes_list}

### Edges (data handoffs)
{edges_list}

### Topological Order
{topological_order}

---

## All Paths to Test (must reach 100% coverage)

{all_paths_display}

---

## Suggested Workflow (reference only — adapt to actual files)

{suggested_workflow}

---

## Current Exploration State

```json
{current_state_json}
```

---

## Skills & Documentation

{skills_docs}

---

## Available Files

{file_summaries}

---

## Output Directory

{output_dir}

---

## Exploration Protocol

### How to Explore

1. **Follow topological order**: Start with source skills (no incoming edges) and work forward.
2. **For EACH edge A->B**:
   a. Use skill A to produce an output artifact
   b. Feed that artifact as input to skill B
   c. Verify the handoff works correctly
   d. Record the edge as tested
3. **Test complete paths end-to-end**: After individual edges work, chain them into full source-to-sink paths.
4. **Core functions**: For each skill, identify and test the functions needed by the DAG workflow.
5. **Other functions**: Explore additional functions for a more complete understanding (lower priority).

### Checkpoint Protocol (CRITICAL)

After approximately every **{checkpoint_interval} tool calls**, you MUST:

#### 1. Update `{output_dir}/dag_exploration_state.json`

```json
{{{{
  "dag_task_id": "{dag_task_id}",
  "skill_names": {skill_names_json},

  "all_paths": {all_paths_json},
  "tested_paths": [<paths you have tested end-to-end>],

  "all_edges": {all_edges_json},
  "tested_edges": [<edges you have verified>],

  "core_functions": {{{{<skill: [funcs needed by workflow]>}}}},
  "tested_core_functions": {{{{<skill: [tested funcs]>}}}},
  "other_functions": {{{{<skill: [other discovered funcs]>}}}},
  "tested_other_functions": {{{{<skill: [tested other funcs]>}}}},

  "total_steps": <current_total>,
  "checkpoint_count": <incremented>,
  "exploration_complete": false,
  "completion_reason": ""
}}}}
```

#### 2. Write checkpoint file `{output_dir}/checkpoint_{{checkpoint_count}}.md`

```markdown
# Checkpoint N - Steps X to Y

## Edges Tested This Chunk
- A -> B: (result description)

## Paths Tested This Chunk
- [A, B, C]: end-to-end (success/failure)

## Core Functions Tested
- [skill_name] function(params) -> result

## Coverage Update
- Path coverage: X/Y (Z%)
- Edge coverage: X/Y (Z%)
- Core function coverage: X/Y (Z%)

## Next Focus
- Which paths/edges still need testing
```

### Termination Conditions

Set `exploration_complete: true` when ALL of these are met:
1. **Path Coverage = 100%**: Every source-to-sink path has been tested end-to-end
2. **Core Function Coverage >= 90%**: Core functions for each skill (needed by DAG workflow) have been tested

When setting complete, provide a clear `completion_reason`.

### DO NOT
- Set `exploration_complete: true` before 100% path coverage
- Skip the checkpoint protocol
- Test paths out of topological order (source skills must work before downstream ones)

### DO
- Update the state file after every ~{checkpoint_interval} tool calls
- Build on previous checkpoints (read them if resuming)
- Prioritize untested paths and edges
- Document concrete examples of data handoffs

---

## Final Output

When `exploration_complete: true`, write `{output_dir}/exploration_summary.md`:
- All skills explored and their key functions
- Verified data handoffs (edge-by-edge)
- Complete path test results
- Known limitations and edge cases
- Concrete examples of data flowing through the DAG

This summary will be used for subsequent task generation.

---

{resume_instruction}

Now begin (or continue) your systematic DAG exploration.
"""

DAG_EXPLORATION_RESUME = """
## RESUMING EXPLORATION (Chunk #{chunk_index})

Your prior work is preserved in:
- `{output_dir}/dag_exploration_state.json` — current progress
- `{output_dir}/checkpoint_*.md` — previous notes

**MUST DO**:
1. Read the state file FIRST to understand what paths/edges are already tested
2. Focus on UNTESTED paths and edges
3. Do NOT repeat tests already recorded in `tested_paths` or `tested_edges`

**Path coverage target: 100%**. Current coverage is below target, which is why exploration is resuming.
"""

FALLBACK_DAG_SUMMARY_PROMPT = """Based on the DAG exploration state and checkpoint files, generate a comprehensive exploration summary.

## DAG Structure
{dag_structure}

## Exploration State
```json
{exploration_state_json}
```

## Checkpoint Files Content
{checkpoint_files_content}

---

## Task

Generate a comprehensive `exploration_summary.md` that includes:

1. **DAG Structure**: Description of the skill dependency graph
2. **Skills Explored**: For each skill, functions discovered and tested
3. **Edge Verification**: For each edge, how data flows from source to target
4. **Path Test Results**: End-to-end path test outcomes
5. **Coverage Summary**: Path, edge, and function coverage percentages
6. **Key Findings**: Important patterns and data handoff mechanisms
7. **Known Limitations**: Edge cases or issues discovered

Format as a complete markdown document.
"""
