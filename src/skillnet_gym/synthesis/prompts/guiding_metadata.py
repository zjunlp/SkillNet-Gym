"""Prompt for generating guiding metadata (execution hints) for task completion"""

SYS_PROMPT = """You are an expert at creating step-by-step execution guidance for AI agents.
Based on a task instruction and skill exploration findings, generate detailed hints to ensure correct task execution.

Your guidance should:
1. Be actionable and specific
2. Reference actual functions and workflows discovered during exploration
3. Anticipate common pitfalls and provide solutions
4. Include verification criteria for each major step"""

USER_PROMPT = """# Input

## Task Instruction
{task_instruction}

## Exploration Summary
<exploration_summary>
{exploration_summary}
</exploration_summary>

## Available Skills
{skills}

## Target Files
{files}
{dag_workflow_section}
---

# Guiding Metadata Generation Guidelines

Generate structured "guiding metadata" that helps an AI agent execute the task correctly.
The guidance MUST be based on the exploration summary's discovered workflows and functions.

## Required Sections

### 1. Key Insights
Identify the most relevant information from the exploration summary for this specific task:
- Which workflows from the exploration are most applicable
- File-specific characteristics based on content type (form/text/table/etc.)
- Critical constraints or edge cases discovered during exploration
- Library/function recommendations from the exploration

### 2. Recommended Approach
Provide a concrete step-by-step execution strategy:
- Reference specific functions and their usage patterns from the exploration
- Include code snippets or command examples where helpful
- Specify expected intermediate outputs at each step
- Note any dependencies between steps

### 3. Pitfalls to Avoid
List potential issues based on exploration findings:
- Known limitations mentioned in the exploration summary
- Error scenarios and how to handle them
- Common mistakes for this type of task
- Edge cases that need special handling

### 4. Verification Criteria
Define how to validate correctness:
- Checks to perform after each major step
- Final output validation requirements
- Self-verification loop mechanism
- Success indicators

---

# Output Format

## Guiding Metadata

### Key Insights
- [Insight 1 based on exploration findings]
- [Insight 2 about file characteristics]
- [Insight 3 about applicable workflows]

### Recommended Approach
1. **Step 1: [Action]**
   - Use: `function_name(params)` from [library]
   - Expected output: [description]

2. **Step 2: [Action]**
   - Use: `another_function(params)`
   - Expected output: [description]

[Continue as needed...]

### Pitfalls to Avoid
- **[Pitfall 1]**: [Description and solution]
- **[Pitfall 2]**: [Description and solution]

### Verification Criteria
- [ ] [Check 1]
- [ ] [Check 2]
- [ ] [Final validation check]

---

# Important Notes

1. Only recommend functions/workflows that appear in the exploration summary
2. Be specific - reference actual function names, libraries, and code patterns
3. Adapt guidance to the specific content type of the target files
4. Include error handling recommendations from the exploration's "Known Limitations" section
"""
