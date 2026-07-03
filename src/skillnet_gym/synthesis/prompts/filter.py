instruction_filter = """You are an expert evaluator of task instructions. Your job is to score a given task instruction across several dimensions of clarity, testability, and naturalness.

Evaluate the instruction on the following 6 dimensions, each on a scale from 0 to 10:

1. Goal Clarity
Definition: Whether the task objective is explicit, concrete, and unambiguous rather than vague.
Common problems:
- vague action verbs
- missing deliverable specification
- undefined success criteria
- ambiguous scope
Scoring guide:
- 0–2: objective is highly vague or unclear
- 3–4: major ambiguity remains
- 5–6: partially clear but still underspecified
- 7–8: mostly clear and actionable 
- 9–10: highly specific, concrete, and unambiguous

2. Input Clarity
Definition: Whether the required input, source material, system, file, version, or reference is clearly specified.
Common problems:
- unspecified data source, file, system, or version
- ambiguous qualifiers such as “latest”
- unclear which input should be used
Scoring guide:
- 0–2: essential input is missing or unclear
- 3–4: major input ambiguity
- 5–6: input is partially defined
- 7–8: input is mostly clear
- 9–10: input is fully and precisely specified

3. Constraint Completeness
Definition: Whether the instruction clearly specifies constraints, thresholds, filters, limits, timelines, exclusion criteria, formatting rules, and precision requirements.
Common problems:
- missing thresholds or limits
- missing timeline or deadline
- missing exclusions or filters
- vague precision requirements
Scoring guide:
- 0–2: constraints are largely missing
- 3–4: many important constraints are absent
- 5–6: some constraints are present but incomplete
- 7–8: constraints are mostly complete
- 9–10: constraints are comprehensive and precise

4. Referential Clarity
Definition: Whether all entities, actors, recipients, tools, pronouns, and domain references are clearly identifiable.
Common problems:
- missing actor or recipient
- referent ambiguity (“it”, “they”, “this”)
- unclear tool, platform, or domain context
Scoring guide:
- 0–2: severe referential ambiguity
- 3–4: major confusion about actors or references
- 5–6: some ambiguity remains
- 7–8: mostly clear references
- 9–10: all references are explicit and unambiguous

5. Verifiability / Uniqueness of Evaluation
Definition: Whether the task output can be objectively and uniquely verified, ideally through deterministic checks such as unit tests, exact-match criteria, structured validation, or clearly bounded correctness conditions.
This is the most important dimension.
Key principle:
- High score: there is a clearly testable correct/incorrect criterion, or a tightly bounded expected output.
- Low score: the task is open-ended, subjective, creative, or allows many equally valid outputs.
Examples:
- High verifiability: “Write a Python function that returns the Fibonacci number for input n and passes the following pytest tests...”
- Low verifiability: “Summarize this article”, “Write a persuasive email”, “Brainstorm product ideas”
Scoring guide:
- 0–2: not objectively verifiable; highly open-ended
- 3–4: weak verification; many acceptable outputs
- 5–6: partially verifiable but still broad
- 7–8: mostly testable with limited ambiguity
- 9–10: strongly and uniquely verifiable with deterministic or near-deterministic evaluation

6. Human-Likeness of the Instruction
Definition: Whether the instruction sounds like something a real human would naturally ask, rather than an artificial, awkward, or machine-generated prompt.
Focus on:
- semantic coherence
- natural task logic
- realistic human intent
- normal phrasing and instruction style
Scoring guide:
- 0–2: highly unnatural or nonsensical
- 3–4: noticeably artificial
- 5–6: somewhat plausible but awkward
- 7–8: mostly natural and human-like
- 9–10: very natural, realistic, and human-like

Weighting:
Use the following weights when computing the final weighted average score:
- Goal Clarity: 0.15
- Input Clarity: 0.15
- Constraint Completeness: 0.15
- Referential Clarity: 0.10
- Verifiability / Uniqueness of Evaluation: 0.20
- Human-Likeness of the Instruction: 0.20

Instructions for evaluation:
- Be strict and discriminative in scoring.
- Do not give high scores unless the instruction truly deserves them.
- Penalize any ambiguity that would prevent a model or programmer from executing or validating the task reliably.
- For verifiability, prioritize whether the output can be uniquely checked, ideally by code, tests, schemas, exact criteria, or tightly constrained expectations.
- If the task is inherently open-ended, subjective, or admits many reasonable outputs, score verifiability low even if the instruction is otherwise clear.

Return your answer in the following JSON format only:

{
  "goal_clarity": {
    "score": <0-10>,
    "reason": "<brief explanation>"
  },
  "input_clarity": {
    "score": <0-10>,
    "reason": "<brief explanation>"
  },
  "constraint_completeness": {
    "score": <0-10>,
    "reason": "<brief explanation>"
  },
  "referential_clarity": {
    "score": <0-10>,
    "reason": "<brief explanation>"
  },
  "verifiability_uniqueness": {
    "score": <0-10>,
    "reason": "<brief explanation>"
  },
  "human_likeness": {
    "score": <0-10>,
    "reason": "<brief explanation>"
  },
  "weighted_average": {
    "score": <0-10>,
    "formula": "0.15*goal_clarity + 0.15*input_clarity + 0.15*constraint_completeness + 0.10*referential_clarity + 0.20*verifiability_uniqueness + 0.20*human_likeness"
  }
}

Now evaluate the following instruction:

[INSERT TASK INSTRUCTION HERE]
"""


dag_compliance_filter = """You are an expert evaluator of DAG-constrained task instructions. A task was generated from a directed acyclic graph (DAG) of skill dependencies. Your job is to evaluate whether the generated instruction properly respects the DAG structure.

## DAG Structure

**Structure Type**: {structure_type}

**Skill Nodes** (all must be necessary in the task):
{nodes_list}

**Edges** (data handoffs that must be reflected):
{edges_list}

**Topological Order**: {topological_order}

---

## Evaluation Dimensions

Evaluate the instruction on the following 3 dimensions, each on a scale from 0 to 10:

1. Skill Coverage
Definition: Whether the task implicitly requires ALL skill nodes in the DAG to be used. No skill should be optional or skippable.
Scoring guide:
- 0–2: Most skills are unnecessary for the task
- 3–4: Several skills are not needed
- 5–6: Some skills could be skipped
- 7–8: Nearly all skills are implicitly required
- 9–10: Every skill node is clearly necessary to complete the task

2. Topological Consistency
Definition: Whether the logical flow of the task respects the topological order of the DAG. Upstream skill outputs should feed into downstream skills.
Scoring guide:
- 0–2: Task flow contradicts the DAG order
- 3–4: Major ordering violations
- 5–6: Partially consistent but some steps are out of order
- 7–8: Mostly follows topological order with minor issues
- 9–10: Task flow perfectly matches the DAG's topological ordering

3. Edge Semantics
Definition: Whether the data handoffs described by DAG edges are naturally reflected in the task. The output of each source skill should serve as meaningful input to its target skill.
Scoring guide:
- 0–2: No data flow between skills is reflected
- 3–4: Data handoffs are largely missing or artificial
- 5–6: Some edges are reflected but others are not
- 7–8: Most data handoffs are naturally expressed
- 9–10: All edge semantics are clearly and naturally reflected in the task

---

## Instructions for evaluation

- Be strict: a task that explicitly lists skills or reveals the DAG structure should score LOWER on naturalness (but this dimension is not scored here).
- Focus on whether the task IMPLICITLY requires the DAG workflow without explicitly naming it.
- A high skill_coverage score means the task cannot be solved without involving every skill.
- A high topological_consistency score means someone reading the task would naturally execute skills in the DAG order.

Return your answer in the following JSON format only:

{{
  "skill_coverage": {{
    "score": <0-10>,
    "reason": "<brief explanation>"
  }},
  "topological_consistency": {{
    "score": <0-10>,
    "reason": "<brief explanation>"
  }},
  "edge_semantics": {{
    "score": <0-10>,
    "reason": "<brief explanation>"
  }},
  "dag_weighted_average": {{
    "score": <0-10>,
    "formula": "0.40*skill_coverage + 0.30*topological_consistency + 0.30*edge_semantics"
  }}
}}

---

Now evaluate the following instruction against the DAG structure above:

{task_instruction}
"""