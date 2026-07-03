"""Prompts for expectation test generation (Phase A: before oracle execution)"""

# =============================================================================
# 验证清单提取 Prompt
# =============================================================================

VERIFICATION_CHECKLIST_SYSTEM = """You are an expert at analyzing task instructions and extracting verification requirements.

Your job is to identify what needs to be verified after a task is completed, based on the instruction and available skill capabilities.

Focus on:
1. Output file paths and formats
2. Required content/fields in outputs
3. Data constraints (types, ranges, non-empty)
4. Semantic requirements (what the output should represent)

Output a structured JSON checklist that can be used to generate pytest tests."""

VERIFICATION_CHECKLIST_USER = """## Task Instruction
{instruction}

## Exploration Summary (Skill Capabilities)
{exploration_summary}

## Input File Summary
{file_summary}

---

Based on the above information, extract a verification checklist in the following JSON format:

```json
{{
  "output_files": [
    {{
      "path": "/root/output/filename.ext",
      "format": "json|csv|md|txt|xlsx",
      "description": "What this file should contain"
    }}
  ],
  "required_content": [
    {{
      "file": "filename.ext",
      "check_type": "contains_key|has_rows|not_empty|matches_pattern|is_list|is_dict",
      "params": {{}},
      "description": "What this check verifies"
    }}
  ],
  "data_constraints": [
    {{
      "file": "filename.ext",
      "field": "field_name",
      "constraint": "type_is|range|min_length|not_null",
      "params": {{}},
      "description": "Constraint description"
    }}
  ],
  "semantic_checks": [
    {{
      "description": "High-level semantic requirement",
      "verification_hint": "How to verify this"
    }}
  ]
}}
```

Important:
- Be specific about expected output file paths (use /root/output/ as default)
- Include ALL files mentioned in the instruction
- For JSON outputs, identify expected keys if possible
- For CSV outputs, identify expected columns if possible
- Include checks for data validity (non-empty, correct types)

Respond with ONLY the JSON, no explanation."""


# =============================================================================
# 测试代码生成 Prompt
# =============================================================================

GENERATE_TESTS_SYSTEM = """You are an expert Python test engineer. Generate pytest test code based on a verification checklist.

Requirements:
1. Use standard pytest conventions
2. Include clear docstrings for each test
3. Use appropriate assertions with helpful error messages
4. Handle file reading errors gracefully
5. Tests should be independent and self-contained"""

GENERATE_TESTS_USER = """## Verification Checklist
```json
{checklist_json}
```

## Output Directory
{output_dir}

---

Generate a complete pytest test file (test_expectations.py) that verifies all items in the checklist.

The test file should:
1. Import necessary modules (os, json, csv, pytest)
2. Define a TestExpectations class
3. Include tests for:
   - File existence
   - File format validity
   - Required content/fields
   - Data constraints
4. Use descriptive test names and docstrings
5. Include helpful assertion messages

Example test structure:
```python
import json
import os
import pytest


class TestExpectations:
    \"\"\"Expectation tests generated from task instruction.\"\"\"

    def test_output_file_exists(self):
        \"\"\"Verify output file was created.\"\"\"
        assert os.path.exists("/root/output/result.json"), "Output file not found"

    def test_output_is_valid_json(self):
        \"\"\"Verify output is valid JSON format.\"\"\"
        with open("/root/output/result.json") as f:
            data = json.load(f)
        assert data is not None
```

Generate ONLY the Python code, no explanation."""


# =============================================================================
# 测试审查补充 Prompt
# =============================================================================

TEST_REVIEW_SYSTEM = """You are a QA expert reviewing test coverage. Your job is to identify missing test cases and suggest improvements.

Focus on:
1. Are all output files tested for existence?
2. Are all format requirements validated?
3. Are all content requirements checked?
4. Are edge cases covered?
5. Are error messages helpful?"""

TEST_REVIEW_USER = """## Task Instruction
{instruction}

## Generated Tests
```python
{generated_tests}
```

---

Review these tests for completeness against the task instruction.

If there are missing tests, respond in this format:
```json
{{
  "coverage_complete": false,
  "missing_tests": [
    {{
      "test_name": "test_something",
      "description": "What this test should verify",
      "code": "def test_something(self):\\n    ..."
    }}
  ],
  "improvements": [
    {{
      "existing_test": "test_name",
      "suggestion": "How to improve"
    }}
  ]
}}
```

If tests are complete, respond:
```json
{{
  "coverage_complete": true,
  "missing_tests": [],
  "improvements": []
}}
```

Respond with ONLY the JSON, no explanation."""


# =============================================================================
# 测试代码合并 Prompt
# =============================================================================

MERGE_TESTS_TEMPLATE = '''"""Auto-generated expectation tests for task verification.

These tests are generated BEFORE oracle execution to verify the task was completed correctly.
"""

import json
import os
from pathlib import Path

import pytest


class TestExpectations:
    """Expectation tests generated from task instruction."""

{test_methods}
'''


# =============================================================================
# 辅助函数
# =============================================================================

def format_file_summary(file_summary) -> str:
    """Format FileSummaryResult for prompt."""
    if not file_summary or not hasattr(file_summary, 'files'):
        return "[No file summary available]"

    lines = []
    for entry in file_summary.files:
        lines.append(f"- **{entry.name}** ({entry.content_type})")
        lines.append(f"  Path: {entry.path}")
        lines.append(f"  Summary: {entry.summary}")
        lines.append("")

    return "\n".join(lines) if lines else "[No files]"


# =============================================================================
# Claude Code 执行器测试生成 Prompt
# =============================================================================

GENERATE_TESTS_WITH_EXECUTOR_SYSTEM = """You are a test engineer. Your task is to generate pytest verification tests for a given task.

You have access to tools to read files and write code. Use them to:
1. Read the input files to understand their structure
2. Generate appropriate pytest tests based on the task instruction and file content
3. Write the test file to the specified location

Important guidelines:
- Tests must verify that the task output matches the instruction requirements
- Read the actual input files to understand their schema/structure
- Generate tests that check file existence, format validity, and content requirements
- Use clear, descriptive test names and docstrings
- Ensure all generated Python code is syntactically correct"""

GENERATE_TESTS_WITH_EXECUTOR_USER = """## Task Instruction
{instruction}

## Exploration Summary (Skill Capabilities)
{exploration_summary}

## Input Files (Read these to understand structure)
{input_files_list}

## Expected Output Directory
{output_dir}

## Test File Path
{test_file_path}

---

**Your steps:**

1. **Read the input files** using the Read tool to understand their structure:
   - For JSON: identify keys, nested structure
   - For CSV/Excel: identify column names, data types
   - For PDF/documents: understand the content being processed

2. **Analyze the task instruction** to determine:
   - What output files should be created
   - What format they should be in
   - What content/fields are required
   - Any specific values that should be present

3. **Generate pytest tests** that verify:
   - Output files exist at expected paths
   - Output format is valid (valid JSON, non-empty CSV, etc.)
   - Required fields/columns are present
   - Data constraints are satisfied
   - Any specific values mentioned in the instruction

4. **Write the test file** to {test_file_path}

**Test code template:**
```python
\"\"\"Auto-generated expectation tests for task verification.

These tests verify that the task execution produces correct outputs
based on the instruction requirements.
\"\"\"

import json
import os
from pathlib import Path

import pytest


class TestExpectations:
    \"\"\"Expectation tests generated from task instruction.\"\"\"

    def test_output_file_exists(self):
        \"\"\"Verify output file was created.\"\"\"
        assert os.path.exists("/root/output/result.json"), "Output file not found"

    def test_output_is_valid_json(self):
        \"\"\"Verify output is valid JSON format.\"\"\"
        with open("/root/output/result.json") as f:
            data = json.load(f)
        assert data is not None

    # Add more tests based on instruction requirements...
```

Now, please read the input files and generate comprehensive tests."""


# =============================================================================
# 语法修复 Prompt
# =============================================================================

FIX_SYNTAX_ERROR_SYSTEM = """You are a Python expert. Fix syntax errors in the provided code.
Return ONLY the corrected Python code, no explanations."""

FIX_SYNTAX_ERROR_USER = """The following Python code has a syntax error:

```python
{code}
```

Error message: {error}

Please fix the syntax error and return the complete corrected code.
Return ONLY the Python code, no explanations or markdown."""


# =============================================================================
# Proposer 审查 Prompts (Phase 3.3.5)
# =============================================================================

PROPOSER_REVIEW_SYSTEM = """You are a senior QA engineer reviewing pytest test code.

Your task is to review generated tests and determine if they adequately verify the task requirements.

Evaluate:
1. **Completeness**: Are all expected outputs and requirements tested?
2. **Strictness**: Are the test conditions strict enough? (e.g., checking specific values vs just existence)
3. **Edge cases**: Are important edge cases covered?
4. **Error messages**: Are assertion messages descriptive?

Respond with a JSON object indicating whether the tests are acceptable or need improvement."""

PROPOSER_REVIEW_USER = """## Task Instruction
{task_instruction}

## Input Files Summary
{input_files_summary}

## Generated Pytest Code
```python
{generated_tests}
```

---

Review the tests and respond with:
```json
{{
  "approved": true/false,
  "issues": [
    {{
      "category": "completeness|strictness|edge_case|clarity",
      "description": "What is missing or needs improvement",
      "suggestion": "Specific suggestion for improvement"
    }}
  ],
  "summary": "Brief summary of review result"
}}
```

If approved=true, issues should be empty.
If approved=false, provide specific, actionable issues."""

PROPOSER_REFINE_PROMPT = """You previously generated pytest tests, but a review found the following issues:

## Review Feedback
{proposer_feedback}

## Previous Test Code
```python
{previous_tests}
```

Please improve the tests by addressing the issues above. Write the improved test file to {test_file_path}.

Focus on:
1. Adding any missing test cases
2. Making test conditions more strict where needed
3. Improving assertion messages
4. Keeping all existing valid tests"""
