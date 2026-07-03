"""Prompts for Process Reward Model (PRM) trajectory validation.

This module contains prompts used to validate oracle execution trajectories
by calling Claude Code as a process reward model. The PRM can use available
skills to verify trajectory correctness.
"""

# =============================================================================
# PRM Validation System Prompt
# =============================================================================

PRM_VALIDATION_SYSTEM = """You are a Process Reward Model (PRM) validating an oracle execution trajectory.

Your task is to verify that the oracle's execution is correct by:
1. Checking that all file paths exist and are accessible
2. Verifying the computation logic matches the task requirements
3. Reading output files and validating their contents are reasonable

You have access to tools and skills that allow you to:
- Read files and verify their contents
- Run Python code to cross-check computations
- Validate data formats and structures

Be thorough but fair - minor formatting differences are acceptable, but logic errors or wrong values are not."""

# =============================================================================
# PRM Validation User Prompt
# =============================================================================

PRM_VALIDATION_PROMPT = """## Task Instruction
{task_instruction}

## Oracle Execution Trajectory
The following is the trajectory of actions taken by the oracle to complete the task:

{trajectory}

## Available Skills
Skills directory: {skills_hint}

You can use these skills to verify the oracle's work.

---

## Your Validation Task

Please validate this oracle execution by performing the following checks:

### 1. Path Verification
- Check that all input files referenced in the trajectory exist
- Check that all output files were created successfully
- Verify file paths are consistent (no workspace path mismatches)

### 2. Logic Verification
- Review the computation steps in the trajectory
- Verify the logic matches what the task instruction requires
- Check for any obvious errors in the approach

### 3. Output Verification
- Read the output files created by the oracle
- Verify the output format matches requirements (JSON structure, CSV columns, etc.)
- If possible, cross-check computed values by running independent calculations

### 4. Consistency Check
- Verify that input data → processing → output makes logical sense
- Check that no data was lost or corrupted in processing

---

## Output Format

After completing your verification, output your result in EXACTLY this JSON format:

```json
{{
  "is_valid": true,
  "issues": [],
  "feedback": ""
}}
```

OR if there are problems:

```json
{{
  "is_valid": false,
  "issues": [
    "Issue 1: Description of first problem",
    "Issue 2: Description of second problem"
  ],
  "feedback": "Detailed feedback explaining what went wrong and how to fix it. This will be provided to the oracle for retry."
}}
```

IMPORTANT:
- Set is_valid to true ONLY if the trajectory is completely correct
- List ALL issues found, not just the first one
- Make feedback actionable - explain exactly what needs to be fixed
- Be specific about file paths, values, or logic that are wrong

Now, please validate the oracle execution:"""


# =============================================================================
# Response Summarization Prompt (for large tool_responses)
# =============================================================================

RESPONSE_SUMMARIZATION_PROMPT = """Summarize the following tool response concisely, preserving:
1. Key file paths mentioned
2. Important values or results
3. Any errors or warnings
4. The overall outcome (success/failure)

Tool Response:
{response}

Provide a concise summary (max 500 tokens):"""


# =============================================================================
# Retry Prompt Template
# =============================================================================

PRM_RETRY_PROMPT_TEMPLATE = """## Previous Attempt Feedback

Your previous attempt to complete this task was validated by a Process Reward Model, which found the following issues:

### Issues Found:
{issues_list}

### Detailed Feedback:
{feedback}

---

## Your Task (Retry Attempt {attempt_number})

Please complete the task again, addressing the issues mentioned above.

{original_prompt}

IMPORTANT:
- Carefully read the feedback above before starting
- Address ALL the issues mentioned
- Double-check your file paths
- Verify your output format matches the requirements
"""
