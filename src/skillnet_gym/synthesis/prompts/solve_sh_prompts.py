"""Prompt templates for solve.sh generation and regeneration.

This module contains prompts used to generate solve.sh scripts that follow
the skillsbench style: heredoc + Python execution pattern.
"""

# =============================================================================
# Skillsbench solve.sh Code Style Guide
# =============================================================================

SKILLSBENCH_SOLVE_STYLE_GUIDE = """
## solve.sh Code Style Guide (skillsbench format)

### Preferred Pattern: Heredoc + Python Execution

Most tasks should use this pattern - write Python code via heredoc, then execute:

```bash
#!/bin/bash
set -e

# Create the solution Python script using heredoc
cat << 'EOF' > /root/solve_task.py
import json
import sys

# Add skill path if using skills
sys.path.append('/root/.claude/skills/<skill-name>/scripts')
from some_tool import SomeTool

# Hardcoded constants are ALLOWED and encouraged for clarity
DENSITY_TABLE = {
    1: 0.10,
    10: 7.85,
    42: 5.55,
}

EXPECTED_OUTPUT_PATH = '/root/output.json'

def main():
    # Read input file
    with open('/root/input.csv') as f:
        data = f.read()

    # Process data - hardcoded values are fine when known
    result = {
        "field1": "computed_value",
        "field2": 123.45,
    }

    # Write output
    with open(EXPECTED_OUTPUT_PATH, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == '__main__':
    main()
EOF

# Execute the script
python3 /root/solve_task.py
```

### Code Style Rules:

1. **Use heredoc with single quotes** ('EOF') to prevent shell variable expansion
2. **Hardcoded values are ALLOWED** - use them when values are known from the trajectory
   - Hardcode computed results if computation is complex
   - Hardcode lookup tables, constants, expected outputs
3. **All paths use /root/ prefix** - input files at /root/<filename>, output at /root/<output>
4. **Import skills if needed** - add sys.path.append for skill scripts
5. **Define constants at top** - makes script self-contained and readable
6. **Use main() function** - clean structure, easy to debug

### Alternative: Direct Shell Commands (for simple tasks)

For simple file creation tasks, direct heredoc is fine:
```bash
#!/bin/bash
set -e

cat << 'EOF' > /root/answer.txt
The answer is 42.
EOF
```

### What NOT to do:

- DO NOT import from test files (expectation_tests.py, test_*.py)
- DO NOT use relative paths (./input.csv) - always use /root/
- DO NOT read output files to verify - just create them correctly
- DO NOT use inline python -c for complex code - use heredoc + file
"""

# =============================================================================
# Regeneration System Prompt
# =============================================================================

REGENERATE_SOLVE_SH_SYSTEM = """You are an expert Shell script engineer.
Your task is to generate a solve.sh script that will pass the given tests.
The script will be executed in a clean Linux environment containing the input files.

""" + SKILLSBENCH_SOLVE_STYLE_GUIDE + """

IMPORTANT: Hardcoded values are perfectly acceptable when they come from the trajectory!
"""

REGENERATE_SOLVE_SH_USER = """## Task Instruction
{task_instruction}

## Execution Trajectory Summary
{trajectory_summary}

## Test Failures
{test_failures}

## Previous Failed solve.sh
```bash
{previous_solve_sh}
```
{constants_section}
Please generate a new solve.sh script that:
1. Creates all required output files
2. Ensures output file contents meet the task instruction requirements
3. Fixes the issues in the previous solve.sh
4. Is COMPLETELY SELF-CONTAINED - no imports from test files
5. Defines all constants and values directly in the script (use the values from the Constants Reference above if provided)

CRITICAL: Do NOT import anything from expectation_tests.py or any test file.
All values, constants, and data must be embedded directly in the script.

Output ONLY the solve.sh content without ```bash``` markers.
"""

# Constants extraction section (appended when test file content is available)
CONSTANTS_REFERENCE_SECTION = """
## Constants Reference (from expectation_tests.py)
The previous solve.sh tried to import values from the test file. Here are the actual constant definitions you should use inline:

```python
{constants_content}
```

Extract the relevant constant values and define them directly in your Python code within solve.sh.
"""

# Initial solve.sh generation prompt (when trajectory-based generation fails)
GENERATE_SOLVE_SH_SYSTEM = """You are an expert Shell script engineer.
Your task is to generate a solve.sh script that reproduces the output of a successful task execution.
The script will be executed in a clean Linux environment containing the input files.

IMPORTANT CONSTRAINTS:
- The script must be COMPLETELY SELF-CONTAINED
- NEVER import from test files (e.g., expectation_tests.py, test_*.py)
- All constants, data, and values must be defined WITHIN the script itself
- Use heredocs for creating files with content
- For Python code, embed all required values directly in the code
"""

GENERATE_SOLVE_SH_USER = """## Task Instruction
{task_instruction}

## Execution Trajectory Summary
The task was completed successfully with the following operations:
{trajectory_summary}

## Output Files to Create
{output_files}

Please generate a solve.sh script that:
1. Creates all the required output files listed above
2. Produces output content that matches the task requirements
3. Uses standard shell commands (bash, cat, echo, python3, etc.)
4. Is COMPLETELY SELF-CONTAINED - no imports from test files
5. Defines all constants and values directly in the script

CRITICAL: Do NOT import anything from expectation_tests.py or any test file.
All values, constants, and data must be embedded directly in the script.

Output ONLY the solve.sh content without ```bash``` markers.
"""

# ============================================================================
# Claude Code executor-based solve.sh generation prompts
# ============================================================================

GENERATE_SOLVE_WITH_EXECUTOR_SYSTEM = """You are an expert at creating shell scripts that reproduce file processing tasks.

""" + SKILLSBENCH_SOLVE_STYLE_GUIDE + """

You have access to:
1. The trajectory summary showing what operations were performed
2. The input files that were processed
3. Skills that can be imported if needed

Your task is to write a solve.sh script that recreates the output files.
IMPORTANT: Hardcoded values are perfectly acceptable when they come from the trajectory!
"""

GENERATE_SOLVE_WITH_EXECUTOR_USER = """## Task Instruction (CRITICAL - MUST FOLLOW EXACTLY)
{task_instruction}

## CRITICAL OUTPUT REQUIREMENTS:
1. The output file paths in your solve.sh MUST EXACTLY match what the Task Instruction specifies
2. The output JSON/data schema MUST EXACTLY match the schema described in the Task Instruction
3. DO NOT invent different file names or change the expected output structure
4. After path normalization, use /root/output/<filename> or /root/<filename> as appropriate
5. If the instruction specifies a JSON schema with specific field names, use EXACTLY those names

## Trajectory Summary (how the oracle completed the task)
{trajectory_summary}

## Input Files (available at /root/)
{input_files_list}

## Skills Directory
Skills are available at: /root/.claude/skills/

---

Write a solve.sh script following the skillsbench style guide.
- Use heredoc to write Python code to /root/solve_task.py
- **CRITICAL**: Output paths and JSON schema MUST match the Task Instruction requirements exactly
- Hardcode values from the trajectory when appropriate
- Execute with python3 /root/solve_task.py
- All output paths should use /root/ prefix (normalized from instruction paths)

Write the script to: {solve_sh_path}
"""

REFINE_SOLVE_WITH_EXECUTOR_PROMPT = """The solve.sh you generated failed verification.

## Test Failures
{test_failures}

## Previous solve.sh (failed)
```bash
{previous_solve_sh}
```

## Task Instruction (CRITICAL - MUST FOLLOW EXACTLY)
{task_instruction}

## CRITICAL: Check these common issues first:
1. Does the output file path EXACTLY match what the Task Instruction specifies?
2. Does the output JSON schema EXACTLY match the instruction's expected schema?
3. Are all required field names EXACTLY as specified in the instruction?
4. Are numeric values in correct format (real numbers, not strings)?
5. Are list/array structures correct as per the instruction?

## Trajectory Summary
{trajectory_summary}

---

Please fix the solve.sh:
1. **FIRST**: Verify output path and schema match the Task Instruction EXACTLY
2. Analyze why the tests failed
3. Check if output values/paths are correct
4. Hardcode values from the trajectory if computation is complex
5. Ensure all paths use /root/ prefix
6. Follow the skillsbench heredoc + python3 pattern

Write the corrected script to: {solve_sh_path}
"""
