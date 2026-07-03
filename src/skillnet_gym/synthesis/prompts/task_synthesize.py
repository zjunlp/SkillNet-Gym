SYS_PROMPT = "You are a task design expert. Based on skill exploration results, generate a specific, achievable task for an AI agent to execute."

# 多文件任务的额外指令
MULTI_FILE_INSTRUCTION = """
## IMPORTANT: Multi-File Task Requirement

You are provided with **{file_count} input files**. Your task MUST meaningfully use **ALL** of them together.

The files share common characteristics or domain relevance. Design a task that:
- Requires information from **each** input file to complete
- Cross-references, compares, or combines data across files
- Would be incomplete or impossible without any one of the files

Examples of good multi-file task patterns:
- Compare financial data across multiple years/companies
- Cross-reference form fields to verify consistency
- Extract and merge information from related documents
- Identify patterns or differences across similar documents

Do NOT design a task that only uses one file while ignoring the others.
"""

USER_PROMPT = """# Input
- 1. Exploration Summary
<Exploration Summary>
{summary_table}
</Exploration Summary>

- 2. Available Skills
{skills}

- 3. Available Files
{files}

---

# Task Synthesis Objective

Generate exactly one task instruction that is:

- complex primarily because it requires reasoning over the semantic content of the available file(s),
- fully solvable using only the available skills and files,
- guaranteed to have a uniquely verifiable final answer,
- clearly grounded in the specific document content rather than generic file-format processing,
- natural and coherent as a real user request about this particular file.

The task should feel like a single meaningful challenge, not a checklist of unrelated analyses.

---

# Task Synthesis Guidelines

## 1) Core goal
- The task must have exactly one primary objective.
- If multiple steps are required, they must all contribute directly to the same final objective.
- Complexity must come from multi-step dependency, cross-reference, computation, filtering, or transformation — not from asking for many unrelated outputs.
- The primary objective must be about the document’s substantive content or purpose, not about its raw file structure.

## 2) Output design
- Prefer a single final deliverable whenever possible.
- The final result must be delivered as a file artifact, not as console/stdout output or only in the final response.
- The task must explicitly specify the exact output file path for the required deliverable.
- Make clear that writing the file is mandatory for task completion.
- Do not use ambiguous wording such as "output", "provide", or “return” unless it is explicitly tied to saving a file artifact.
- Prefer minimal file deliverables that are sufficient for verification.
- Prefer conclusion-oriented outputs over analysis-oriented outputs.

## 3) Verifiability and uniqueness
- The final answer must be deterministic and uniquely checkable by test code.
- Favor task styles whose correctness can be validated automatically, such as:
  - a single numeric result,
  - a uniquely determined string / identifier / record,
  - a small JSON object with a fixed schema and fixed semantics,
  - a transformed file whose expected content is uniquely derivable.
- If intermediate reasoning is needed, do not require dumping all intermediate results unless they are necessary for verification.
- Do not satisfy verifiability merely by converting generic extracted artifacts into a checksum or fingerprint.
- Prefer tasks whose final answer is uniquely determined by the document content itself.

## 4) Use of exploration summary
- Use the exploration summary to infer logical relationships between available operations and data.
- Do not mechanically convert the exploration summary into a feature checklist.
- Select only the subset of capabilities needed for one coherent task.
- If only one file is provided, focus on that file. If multiple files are provided, design a task that meaningfully uses ALL of them together.

## 5) Naturalness
- The task should read like a realistic user request.
- Avoid “audit/report/inventory/analyze everything” style tasks unless the exploration strongly implies that such a report itself is the single natural end goal.
- Avoid tasks that feel like benchmark instrumentation rather than a genuine problem to solve.

## 6) File constraints
- Only mention input files that appear in 'Available Files'.
- Do not introduce nonexistent resources, hidden metadata, or external dependencies.
- **CRITICAL**: The exploration summary may reference example file names from skill documentation (like input.bam, reads.bam, reference.fa, sample*.bam). These are EXAMPLES in the documentation, NOT actual files! They do NOT exist in the environment.
- You MUST use ONLY the exact file paths listed in the 'Available Files' section below. Do not assume any other files exist.

## 7) Difficulty control
- The task should be nontrivial but reasonably scoped.
- Avoid tasks whose natural solution is to build a generic full parser, full crawler, or full-document auditor unless that is truly required for the unique final answer.
- Prefer targeted extraction / transformation / computation over exhaustive enumeration.

## 8) Anti-patterns to avoid
Do NOT generate tasks that:
- bundle several independent subtasks into one instruction,
- ask for many loosely related outputs,
- request full audits, inventories, dashboards, or exploratory summaries by default,
- require unnecessary debug artifacts,
- include verification-oriented details that do not serve the actual user goal,
- are underdetermined or allow multiple valid final answers.

## 9) Full-skill necessity
- The task must be constructed so that completing it correctly requires using ALL provided skills in combination.
- No proper solution should be possible by relying on only a subset of the available skills.
- Each required skill must make a necessary contribution to the single final objective, rather than being included incidentally.
- The need to use all skills should be implicit in the task design, not stated explicitly as an instruction to 'use all skills.'
- Avoid wording that reveals the benchmark intent or directly tells the model which skills to invoke.

## 10) Self-check before writing
Before producing the task, ensure:
- There is exactly one main question to answer.
- Every required step is necessary for that question.
- The final answer is unique and machine-verifiable.
- The requested output is minimal but sufficient.
- The task is solvable using only the provided files and skills.
- The final artifact is a compact answer, not a broad listing of intermediate facts.

---

# Good Task Characteristics

A good synthesized task usually has the following pattern:
- one clear end goal,
- a small number of tightly coupled steps,
- one final artifact,
- one uniquely derivable answer.

Examples of good task shapes:
- recover missing values by combining information across sheets and save the corrected file;
- identify the single valid object/component/record satisfying derived constraints and output its key attributes;
- compute a final metric from structured extraction + deterministic filtering + reference lookup;
- perform a targeted code migration/change and validate it with a specific build/test command.

---

# Bad Task Characteristics

Avoid tasks shaped like:
- “extract everything, summarize everything, visualize everything”,
- “produce JSON + CSV + image + stdout summary” when one output would suffice,
- “audit the document/codebase and report all metadata”,
- multiple parallel sections (A/B/C) that do not depend on one another.

---

Good Task Instruction Examples

Example 1:
Recover missing values in an Excel file `nasa_budget_incomplete.xlsx`. Missing values are marked with "???". Analyze the relationships between sheets to determine the correct values and replace each "???" with the computed numeric value. Save as `nasa_budget_recovered.xlsx`

Example 2:
Calculate the mass of a 3D printed part. The input (`/root/scan_data.stl`) is a binary STL with Material ID stored in the attribute byte count field.

Parse the binary STL, identify the **largest connected component** (filtering out scanning debris), extract the Material ID, reference `/root/material_density_table.md` for density, and calculate mass using `Volume * Density`.

Save to `/root/mass_report.json`:
```json
{{
  "main_part_mass": 12345.67,
  "material_id": 42
}}
```
NOTE: Result accuracy within 0.1% is acceptable.

Example 3:
A Java repository in /home/travis/build/failed/<repo>/<id> currently fails to build.

Determine the root cause of the failure and fix it by modifying the repository. The root cause may be in the source code or in the build configuration.

Before applying the fix, write a short diagnosis and repair plan to /home/travis/build/failed/failed_reasons.txt. Then save your proposed code changes as standard diff files in /home/travis/build/failed/<repo>/<id>/patch_{{i}}.diff, apply them, and ensure the build succeeds.

Do not make unrelated refactors or feature additions.
---

# Output

Generate the task in the following format (use this exact header):

# Synthesized Task Instruction

"""


