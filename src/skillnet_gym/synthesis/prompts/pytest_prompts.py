"""Prompts for pytest test generation in skillsbench style.

This module contains prompts used to generate pytest tests that verify
task outputs using hardcoded expected values and structural checks.
"""

# =============================================================================
# Pytest Generation System Prompt
# =============================================================================

PYTEST_GENERATION_SYSTEM = """You are a test engineer creating pytest tests for verifying task outputs.

Your tests should follow the skillsbench style:
1. Use hardcoded expected values when appropriate (simple and reliable)
2. Include structural tests (file exists, valid format)
3. Include value tests (exact match or with tolerance)
4. Include content tests (required fields, data integrity)

## Test Categories (all required):

### 1. Structural Tests
- File existence: `test_<filename>_exists()`
- Format validity: `test_<filename>_valid_json()`, `test_<filename>_valid_csv()`, etc.

### 2. Value Tests
- Read the actual output files and extract expected values
- Hardcode these values in EXPECTED_RESULT class variable
- Use `math.isclose()` or TOLERANCE for float comparisons

### 3. Content Tests
- Required fields/keys exist
- Data types are correct
- Values are in expected range/format

## Code Style:
- Use class-based tests (e.g., `class TestOutputs:`)
- Define EXPECTED_RESULT and TOLERANCE as class variables
- Use clear docstrings explaining what each test verifies
- All file paths should use absolute paths starting with /root/
"""

# =============================================================================
# Pytest Generation User Prompt
# =============================================================================

PYTEST_GENERATION_PROMPT = """## Task Instruction (CRITICAL - Tests MUST match this)
{task_instruction}

## CRITICAL: Test Requirements Based on Instruction
You MUST generate tests that verify the output files and schema EXACTLY as specified in the Task Instruction above.

**STRICT RULES:**
1. The test file paths MUST match the paths specified in the Task Instruction (after normalization to /root/)
2. The expected JSON schema/structure MUST match EXACTLY what the instruction describes
3. DO NOT test files that aren't mentioned in the instruction
4. DO NOT invent or modify the expected output schema - use EXACTLY what the instruction specifies
5. If instruction shows a JSON schema with specific field names, test for EXACTLY those fields

## Trajectory Summary (how the task was completed)
{trajectory_summary}

## Final Output Files to Verify
{final_files_info}

**NOTE**: If these files don't match what the instruction expects, prioritize the instruction requirements.

---

## Your Task: Generate pytest Tests

Read the final output files and generate a complete pytest test file that verifies them.

### Steps:
1. Read each output file to understand its content and structure
2. Extract key values that should be verified
3. Write tests covering:
   - File existence
   - Format validity (JSON/CSV/etc)
   - Value correctness (hardcoded expected values)
   - Content integrity (required fields, data types)

### Example Test Style (from skillsbench):
```python
import json
import math
import os

import pytest


class TestOutputs:
    \"\"\"Tests for verifying task outputs.\"\"\"

    EXPECTED_RESULT = {{
        "field1": "expected_value",
        "field2": 123.45,
    }}
    TOLERANCE = 0.001

    def test_file_exists(self):
        \"\"\"Verify output file was created.\"\"\"
        assert os.path.exists("/root/answer.json"), "Output file not found"

    def test_valid_json(self):
        \"\"\"Verify output is valid JSON.\"\"\"
        with open("/root/answer.json") as f:
            json.load(f)

    def test_has_required_fields(self):
        \"\"\"Verify all required fields are present.\"\"\"
        with open("/root/answer.json") as f:
            data = json.load(f)
        assert "field1" in data, "Missing required field: field1"
        assert "field2" in data, "Missing required field: field2"

    def test_values_correct(self):
        \"\"\"Verify output values match expected.\"\"\"
        with open("/root/answer.json") as f:
            data = json.load(f)

        assert data["field1"] == self.EXPECTED_RESULT["field1"], \\
            f"field1 mismatch: expected {{self.EXPECTED_RESULT['field1']}}, got {{data['field1']}}"

        assert math.isclose(data["field2"], self.EXPECTED_RESULT["field2"], rel_tol=self.TOLERANCE), \\
            f"field2 mismatch: expected {{self.EXPECTED_RESULT['field2']}}, got {{data['field2']}}"
```

## Output
Write the complete test file to: {test_file_path}

Generate the tests now:"""

# =============================================================================
# Pytest Regeneration Prompt (after failure)
# =============================================================================

PYTEST_RETRY_PROMPT = """## Previous Test Failed

Your previous test failed when run against the final output files.

### Previous Test Code
```python
{previous_test}
```

### Failure Info
{failure_info}

---

## Fix the Test

**CRITICAL: Check these first:**
1. **Test paths match instruction** - are you testing the paths/filenames specified in the Task Instruction?
2. **Schema matches instruction** - does your expected schema match what the instruction requires?
3. **Field names are exact** - are you using the EXACT field names from the instruction?

**Then check:**
4. **Wrong expected values** - re-read the actual files and extract correct values
5. **Tolerance too strict** - increase tolerance for float comparisons
6. **Missing imports** - ensure all required modules are imported

## Task Instruction (CRITICAL - Tests MUST match this)
{task_instruction}

## Trajectory Summary
{trajectory_summary}

## Final Output Files
{final_files_info}

---

**IMPORTANT**: The tests MUST verify what the Task Instruction requires, not what files happen to exist.
If there's a mismatch between instruction and actual files, prioritize the instruction requirements.

Write the corrected test file to: {test_file_path}

Read the output files again and fix the test:"""

# =============================================================================
# Helper function
# =============================================================================

def format_files_info(files: list[str]) -> str:
    """Format file list for prompt.

    Args:
        files: List of file paths

    Returns:
        Formatted string with file paths
    """
    return "\n".join(f"- {f}" for f in files)
