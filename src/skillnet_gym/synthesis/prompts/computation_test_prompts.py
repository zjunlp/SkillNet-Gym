"""Prompts for computation-based test generation.

This module contains prompts used to generate pytest tests that verify
oracle outputs through independent computation, rather than hardcoded values.
"""

import json

# =============================================================================
# Computation Test Generation System Prompt
# =============================================================================

COMPUTATION_TEST_SYSTEM = """You are an expert test engineer generating verification tests for a completed task.

Your goal is to create pytest tests that INDEPENDENTLY VERIFY the oracle's outputs by:
1. Reading the output files produced by the oracle
2. Computing the expected values independently using the same logic
3. Comparing the oracle's output with your independent computation

CRITICAL RULES:
- NEVER hardcode expected values like `EXPECTED_VALUE = 42` or `EXPECTED_NAME = "foo"`
- ALWAYS write computation code that calculates what the expected value should be
- The computation logic should match what the oracle did, based on the trajectory
- Use appropriate tolerances for floating-point comparisons

You have access to tools to read files and write the test code."""


# =============================================================================
# Computation Test Generation User Prompt
# =============================================================================

COMPUTATION_TEST_PROMPT = """## Task Instruction
{task_instruction}

## Oracle Execution Trajectory
The following shows what the oracle did to complete the task:

{trajectory}

## Input Files
{input_files}

## Output Files Created
{output_files}

---

## Your Task: Generate Computation-Based Verification Tests

Analyze the oracle trajectory and generate pytest tests that:

1. **Read the oracle's output files** to get the values it produced
2. **Independently compute the expected values** using the same logic from the trajectory
3. **Compare and assert** that oracle output matches your independent computation

### Example - BAD (hardcoded values):
```python
# ❌ DON'T DO THIS - hardcoded values that may be wrong
EXPECTED_TANIMOTO = 0.0
EXPECTED_VENDOR = "Acme"

def test_tanimoto():
    with open("/root/output.json") as f:
        data = json.load(f)
    assert data["tanimoto"] == EXPECTED_TANIMOTO  # Hardcoded!
```

### Example - GOOD (computation-based):
```python
# ✅ DO THIS - independent computation
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

def compute_tanimoto(mol1_path: str, mol2_path: str) -> float:
    '''Independently compute Tanimoto similarity using same logic as oracle.'''
    mol1 = Chem.SDMolSupplier(mol1_path, sanitize=True)[0]
    mol2 = Chem.SDMolSupplier(mol2_path, sanitize=True)[0]
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def test_tanimoto_matches_computation():
    '''Verify oracle's Tanimoto output matches independent computation.'''
    # Read oracle output
    with open("/root/output.json") as f:
        oracle_result = json.load(f)["tanimoto"]

    # Independently compute expected value
    expected = compute_tanimoto("/root/mol1.sdf", "/root/mol2.sdf")

    # Verify they match
    assert abs(oracle_result - expected) < 0.001, \\
        f"Oracle gave {{oracle_result}}, but computed {{expected}}"
```

### Test Categories to Generate:

1. **Structural Tests** (always include):
   - File existence checks
   - Format validity (valid JSON, non-empty CSV, etc.)
   - Required fields/columns present

2. **Computation Tests** (main focus):
   - For each computed value in the output, write a test that:
     a. Reads the oracle's output value
     b. Computes what the value should be independently
     c. Asserts they match (with appropriate tolerance)

3. **Consistency Tests** (if applicable):
   - Cross-check relationships between values
   - Verify counts/totals are consistent

---

## Output

Write the complete test file to: {test_file_path}

The test file should:
1. Import all necessary libraries
2. Define helper functions for independent computation
3. Define test class with all verification tests
4. Use clear docstrings explaining what each test verifies

Generate the tests now:"""


# =============================================================================
# Computation Test Regeneration Prompt (after failure)
# =============================================================================

COMPUTATION_TEST_REGENERATION_PROMPT = """## Previous Test Failures

Your previously generated computation tests failed with the following errors:

{failure_summary}

## Original Task
{task_instruction}

## Oracle Trajectory
{trajectory}

---

## Your Task: Fix the Computation Tests

Please regenerate the computation tests, addressing the failures above.

Common issues to check:
1. **Import errors**: Make sure all required libraries are imported
2. **Path errors**: Use the correct file paths from the trajectory
3. **Logic mismatches**: Ensure your computation logic exactly matches the oracle's approach
4. **Data type issues**: Handle type conversions properly
5. **Tolerance issues**: Use appropriate tolerances for floating-point comparisons

Write the corrected test file to: {test_file_path}"""


# =============================================================================
# Structural Test Template (fallback)
# =============================================================================

STRUCTURAL_TEST_TEMPLATE = '''"""Auto-generated structural tests for task verification.

These tests verify file existence, format validity, and required structure.
They do NOT verify specific computed values - those require computation tests.
"""

import json
import os
from pathlib import Path

import pytest


class TestStructure:
    """Structural verification tests - existence, format, required fields."""

{file_existence_tests}

{format_validity_tests}

{structure_tests}
'''


# =============================================================================
# Helper function
# =============================================================================

def format_trajectory_for_prompt(trajectory_steps: list, max_steps: int = 50) -> str:
    """Format trajectory steps for inclusion in prompt.

    Args:
        trajectory_steps: List of step dictionaries
        max_steps: Maximum number of steps to include

    Returns:
        Formatted trajectory string
    """
    if len(trajectory_steps) > max_steps:
        # Include first and last steps, with ellipsis in middle
        first_half = trajectory_steps[:max_steps // 2]
        last_half = trajectory_steps[-(max_steps // 2):]
        steps_to_format = first_half + [{"note": f"... ({len(trajectory_steps) - max_steps} steps omitted) ..."}] + last_half
    else:
        steps_to_format = trajectory_steps

    formatted = []
    for i, step in enumerate(steps_to_format, 1):
        if "note" in step:
            formatted.append(step["note"])
        else:
            tool = step.get("tool", "unknown")
            action = step.get("action", {})
            response = step.get("response", step.get("response_summary", ""))

            formatted.append(f"### Step {i}: {tool}")
            formatted.append(f"Action: {json.dumps(action, ensure_ascii=False)[:500]}")
            if response:
                resp_preview = str(response)[:300]
                if len(str(response)) > 300:
                    resp_preview += "..."
                formatted.append(f"Response: {resp_preview}")
            formatted.append("")

    return "\n".join(formatted)
