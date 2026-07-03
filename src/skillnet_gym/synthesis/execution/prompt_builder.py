"""Prompt builder for guiding Claude Code autonomous execution"""

from ..config import PromptConfig, PromptStyle, CreativityLevel


# File summary prompt template
FILE_SUMMARY_PROMPT = """Use the Read tool to read the file at {file_path} (read only the first {max_read_lines} lines), then produce a concise summary in no more than {max_chars} words.

The summary should include the following key information, as applicable to the file type:

- File type and format (e.g. Excel spreadsheet, PDF document, JSON data)
- Data structure (e.g. number of rows, columns, field names, hierarchical structure)
- Main content overview (e.g. topic, data types, value ranges of key fields)
- Special markers or formatting (e.g. missing value indicators, encoding format, delimiters)
- If only partially read, note that this is a preview of the first {max_read_lines} lines of the file

When reading the file, use: Read(file_path="{file_path}", limit={max_read_lines})

Output only the summary text, with no prefix or explanation.
"""
 
# 请使用 Read 工具读取文件 {file_path}（仅读取前 {max_read_lines} 行），然后生成一个简洁的摘要（不超过 {max_chars} 字）。

# 摘要应包含以下关键信息（根据文件类型选择适用项）：
# - 文件类型和格式（如 Excel表格、PDF文档、JSON数据等）
# - 数据结构（如 行数、列数、字段名、层级结构等）
# - 主要内容概述（如 主题、数据类型、关键字段值范围等）
# - 特殊标记或格式（如 缺失值标记、编码格式、分隔符等）
# - 如果是部分读取，说明这是文件的前 {max_read_lines} 行预览

# 读取文件时请使用: Read(file_path="{file_path}", limit={max_read_lines})

# 直接输出摘要文本，不要添加任何前缀或解释。


# Prompt templates for different styles
PROMPT_TEMPLATES = {
    PromptStyle.MINIMAL: """Please use Skills {skills_hint} and operate the file {file_path}.

Your task:
1. First, read the file to understand its contents
2. Use the appropriate skills, and perform the operations provided within those skills on the given files. 
3. The operations should follow a logical sequence. 
4. The final result should be in a uniquely verifiable form. You may choose one of the following result types, or propose a similarly verifiable format of your own:

- Perform numerical calculations and fill in the results in a specific format, such as JSON
- Extract verifiable information and produce a structured document with fixed fields
- Implement code and run it to generate results
- Compare tabular data across different formats and output structured differences
- Solve programming problems in batch and generate corresponding result files
- ...
5. Create output files in {output_dir} with your analysis results

Execute at most {max_steps} steps.
""".strip(),
# """
# Please read and analyze the file {file_path}.

# Your task:
# 1. First, read the file to understand its contents
# 2. Perform meaningful analysis or data extraction
# 3. Create output files in {output_dir} with your analysis results

# You MUST create at least one output file using the Write tool.
# Execute at most {max_steps} steps.
# """.strip(),

    PromptStyle.DOMAIN_GUIDED: """
You are a {domain} expert. Please perform professional analysis and processing on {file_path}.
Output the analysis results and processed files to {output_dir}.
Execute at most {max_steps} steps.
""".strip(),

    PromptStyle.SKILL_HINTED: """
Please use your skills to process {file_path}.
Hint: {skills_hint}
Output processing results to {output_dir}, at most {max_steps} steps.
""".strip(),

    PromptStyle.GOAL_ORIENTED: """
Please complete the following task:
- Read and understand the content of {file_path}
- Perform meaningful analysis or transformation
- Output structured results to {output_dir}

You can freely decide the specific processing method. At most {max_steps} steps.
""".strip(),
}

# Templates with file summaries (for Phase 0 integration)
PROMPT_TEMPLATES_WITH_SUMMARIES = {
    PromptStyle.MINIMAL: """# Role and Mission

You are a **Skill Explorer** with strong curiosity, systematic thinking, and adaptive learning ability.  
Your mission is to **deeply understand** the skills available in the skill library through structured exploration, including their usage mechanisms, operating patterns, and potential applications.

---

## Input Overview

1. **Skills**: Various skills related to progressive disclosure. You must only explore the following skills: [{skills_hint}].
2. **Available files**: Files may be readable and operable:
{file_summaries}
3. **Output path**: Path to the output file generated during exploration, {output_dir}.
4. **Max exploration steps**: Execute at most {max_steps} steps.
5. **Summary of the Exploration History**: Contains the key workflows discovered, the real-world application scenarios identified, and summaries of the internal states of intermediate output files. If left empty, it indicates this is the first exploration. The Exploration History is as below:
{summary_exploration_history}

### Important Notes Before Exploration

During exploration, if file summaries are provided, you should make full use of them:

- Understand the summary of each provided file so you can use the relevant skills on them more effectively later.
- Do not force the use of `Read`, since some files may be unreadable or too long.

During exploration, if a Summarized Exploration History is provided, you should make full use of it:

- Understand the conclusions already drawn from the exploration history.
- Avoid repeating the exploration of patterns that have already been identified.

---

## Core Principles for Skill Exploration

### 1. Multi-Skill Exploration
- **Relevance first**: Among multiple skills, prioritize the ones that best match the provided files.
- **Explore broadly**: Do not neglect exploring other available skills as well.

### 2. Progressive Exploration
- **Avoid simple repetition**: Do not repeatedly test the same function with identical parameters and in the same order.
- **Explore based on results**: Every next step should be informed by the outcome of the previous one.
- **Go deeper when useful**: If an interesting result appears, investigate the related functionality more thoroughly.

### 3. Context-Aware Decision Making
- **Analyze results carefully**: Interpret the return value of each skill call with attention.
- **Track state**: Maintain an internal record of the current environment state and all acquired information.
- **Think associatively**: Identify relationships between different functions and possible ways to combine them.

---

## Exploration Strategy

### Phase 1: Initial Mapping
1. **Broad scan**: Test representative functions to understand the main categories of functionality.
2. **Identify core function types**: Distinguish between query, action, and configuration functions.
3. **Discover data flow**: Identify which functions produce data and which functions consume it.

### Phase 2: Deep Exploration
1. **Chained exploration**: Use the output of one step as the input to the next.
2. **Boundary testing**: Explore parameter ranges and edge cases.
3. **Combination experiments**: Test meaningful combinations of functions, including those involving other script files and functions referenced in supporting files.

### Phase 3: Pattern Discovery
1. **Identify workflows**: Find recurring sequences of operations.
2. **Construct scenarios**: Consider what real-world problems these API sequences could solve.

---

## Operational Decision Framework

Before choosing the next action, ask yourself:

1. **Use of new information**: What new information did I obtain from the previous step, and how can I use it?
2. **Exploration value**: What new understanding will this action provide?
3. **Avoid redundancy**: Is this action too similar to a previous one?
4. **Prefer depth when appropriate**: Should I continue exploring this area more deeply instead of jumping to something unrelated?

---

## Action Selection Guidelines

- **If the previous step returned data**: Try using it as input to other functions.
- **If the previous step failed**: Diagnose the reason, adjust parameters, or try related functions.
- **If the previous step succeeded**: Explore follow-up actions or parameter variations.
- **If a new type of functionality is discovered**: Temporarily pause other exploration and prioritize testing it.

### Avoid
- ❌ Testing functions in alphabetical or fixed order
- ❌ Ignoring returned data
- ❌ Repeating calls with exactly the same parameters
- ❌ Jumping randomly between unrelated areas
- ❌ Repeating the exploration of patterns that have already been identified in the Summarized Exploration History.

### Encourage
- ✅ Choose actions based on previous results
- ✅ Reuse obtained data as input
- ✅ Investigate interesting patterns in depth
- ✅ Discover logical relationships between functions
- ✅ Discover new patterns

---

## Thinking Before Each Step

Before taking an action, reflect on:

1. **Observation**: What did I learn from the previous step?
2. **Reasoning**: Why am I choosing this action?
3. **Goal**: What do I hope to discover?

---

## Internal State to Maintain

Keep track of the following:

- **Known functions** and their purposes
- **Important returned data** and its possible uses
- **Observed patterns** and workflows
- **Hypotheses and ideas** that still need validation

---

## Overall Goal

Your goal is not to complete one specific task, but to develop a deep and structured understanding of the environment's capabilities, constraints, and potential real-world applications through exploration.  
Each step should make your understanding more complete.

## Final Output Overview

After the exploration is complete, you MUST write a summary to the file `{output_dir}/exploration_summary.md` using the Write tool.

The summary should consolidate the findings of this exploration together with any summaries from previous exploration history. Key information should include:
- Key workflows discovered
- Practical application scenarios identified
- Summary of output files created and their internal states
- Do not contain the following skills' usage: [`simplify`, `loop`, `claude-api`, `update-config`] 

This file will be read and used for subsequent exploration loops.

""".strip(),



## Skill-Driven Exploration Objective

# - **Understand the skill first**: Before actually executing functions within a selected skill, first build an understanding of the full set of functions that skill provides.

# - ❌ Performing tasks whose outputs cannot be uniquely evaluated, such as open-ended summarization or creative generation
# - ✅ Perform actions with objectively verifiable outcomes, such as code execution results or extraction of verifiable information




# """Please use Skills {skills_hint} and operate the following file(s):

# {file_summaries}

# Your task:
# 1. Based on the file summaries above, use the appropriate skills to process the files
# 2. The operations should follow a logical sequence
# 3. The final result should be in a uniquely verifiable form. You may choose one of the following result types, or propose a similarly verifiable format of your own:

# - Perform numerical calculations and fill in the results in a specific format, such as JSON
# - Extract verifiable information and produce a structured document with fixed fields
# - Implement code and run it to generate results
# - Compare tabular data across different formats and output structured differences
# - Solve programming problems in batch and generate corresponding result files
# - ...
# 4. Create output files in {output_dir} with your analysis results

# Execute at most {max_steps} steps.
# """.strip(),

    PromptStyle.DOMAIN_GUIDED: """
You are a {domain} expert. Please perform professional analysis and processing on the following file(s):

{file_summaries}

Output the analysis results and processed files to {output_dir}.
Execute at most {max_steps} steps.
""".strip(),

    PromptStyle.SKILL_HINTED: """
Please use your skills to process the following file(s):

{file_summaries}

Hint: {skills_hint}
Output processing results to {output_dir}, at most {max_steps} steps.
""".strip(),

    PromptStyle.GOAL_ORIENTED: """
Please complete the following task:

{file_summaries}

- Perform meaningful analysis or transformation based on the file contents
- Output structured results to {output_dir}

You can freely decide the specific processing method. At most {max_steps} steps.
""".strip(),
}


# Templates for no-file scenarios
NO_FILE_TEMPLATES = {
    PromptStyle.MINIMAL: """
Please create useful content and save it to {output_dir}.
Execute at most {max_steps} steps.
""".strip(),

    PromptStyle.DOMAIN_GUIDED: """
You are a {domain} expert. Please create professional content in your domain.
Save all outputs to {output_dir}.
Execute at most {max_steps} steps.
""".strip(),

    PromptStyle.SKILL_HINTED: """
Please use your skills to create useful content.
Hint: {skills_hint}
Output results to {output_dir}, at most {max_steps} steps.
""".strip(),

    PromptStyle.GOAL_ORIENTED: """
Please complete a creative task:
- Generate useful structured content
- Output results to {output_dir}

You can freely decide the specific approach. At most {max_steps} steps.
""".strip(),
}

# 带历史总结的探索 prompt
EXPLORATION_WITH_HISTORY_TEMPLATE = """# Role and Mission

You are a **Skill Explorer** continuing your systematic exploration.

---

## Previous Exploration Summary

{previous_summary}

---

## Input Overview

1. **Skills**: [{skills_hint}]
2. **Available files**: {file_summaries}
3. **Output path**: {output_dir}
4. **Max exploration steps**: {max_steps}

---

## Continuation Guidelines

Based on previous exploration:
1. **Avoid repeating** already explored functionality
2. **Focus on** unexplored areas and deeper patterns
3. **Build upon** previous findings
4. **Discover** new combinations and workflows

---

## Current Exploration Goals

{exploration_goals}

Execute your exploration now.
""".strip()

# Goal 驱动执行 prompt
# GOAL_DRIVEN_TEMPLATE = """# Goal-Driven Task Execution

# ## Task Instruction

# {task_instruction}

# ---

# ## Execution Guide

# {execution_guide}

# ---

# ## Context from Exploration

# ### Skills Available
# {skills_summary}

# ### Key Findings from Exploration
# {key_findings}

# ---

# ## Input Files
# {file_summaries}

# ## Output Requirements
# - Output directory: {output_dir}
# - Maximum steps: {max_steps}

# ---

# ## Execution Instructions

# Execute the task according to the instruction above.
# Follow the execution guide to ensure correct results.
# Focus on producing verifiable, structured outputs.
# """.strip()

GOAL_DRIVEN_TEMPLATE = """You are a helpful assistant to complete user's instruction with the provided skills and guidelines.

## Input Overview

1. **Skills**: Various skills related to progressive disclosure. You MUST use the following skills: {skills_hint}.
2. **Available files**: Files may be readable and operable:
{file_summaries}
3. **User's Instruction**: {task_instruction}
4. **Guidelines**: {execution_guide}

---

## Execution Instructions

1. Before executing the task, first make sure you fully understand the provided skills and guidelines.
2. **CRITICAL: You MUST use the scripts/functions from the skills directory to process files.** Do NOT simply read files directly to extract information. For example, if processing a PDF, you must use pypdf, pdfplumber, or the scripts in the skills folder.
3. Follow the user's instruction step by step.
4. If you encounter any environment-related issues during execution, you may install any required packages yourself.
5. **CRITICAL: You MUST save your final output to a file** in {output_dir}. Do not just print results to console.
""".strip()


# Checkpoint-based exploration prompt template (supports multiple skills)
CHECKPOINT_EXPLORATION_PROMPT = """# Checkpoint-Based Multi-Skill Exploration

You are a **Skill Explorer** conducting systematic exploration with checkpoint-based progress tracking.
Your mission is to deeply understand **all available skills** through structured exploration.

{resume_instruction}

---

## Current Exploration State

```json
{current_state_json}
```

## Documented Functions (from SKILL.md files, grouped by skill)

These are the functions documented in each skill's files. Your goal is to test as many as possible across ALL skills:
```json
{documented_functions_json}
```

## Skills to Explore

You must explore the following skills: [{skill_names}]

Skill directory hint: [{skills_hint}]

## Available Files

{file_summaries}

## Output Directory

{output_dir}

---

## Checkpoint Protocol (CRITICAL)

After approximately every **{checkpoint_interval} tool calls**, you MUST:

### 1. Update `{output_dir}/exploration_state.json`

Update the state file with your progress (note: functions are grouped by skill):
```json
{{
  "skill_names": ["{skill_names}"],
  "total_steps": <current_total>,
  "checkpoint_count": <incremented>,

  "documented_functions": {{           // Grouped by skill
    "skill1": ["func1", "func2"],
    "skill2": ["funcA", "funcB"]
  }},
  "discovered_functions": {{           // Grouped by skill
    "skill1": ["new_func"],
    "skill2": []
  }},
  "tested_functions": {{               // Grouped by skill
    "skill1": ["func1"],
    "skill2": ["funcA", "funcB"]
  }},

  "workflows": [
    {{"name": "workflow name", "steps": ["skill1.func1", "skill2.funcA"], "skills": ["skill1", "skill2"], "tested": true/false}}
  ],

  "last_new_discovery_step": <step when you last found something new>,
  "consecutive_no_progress": <0 if progress this chunk, else increment>,

  "exploration_complete": false,      // Set true only when done
  "completion_reason": ""             // Explain why when complete
}}
```

### 2. Write checkpoint file `{output_dir}/checkpoint_{checkpoint_count}.md`

Document this exploration chunk:
```markdown
# Checkpoint {checkpoint_count} - Steps X to Y

## New Discoveries This Chunk
- [skill_name] function: description

## Tests Performed
- [skill_name] function(params) -> result (success/failure)

## Updated Coverage
- Overall: X/Y tested (Z%)
- skill1: A/B tested (C%)
- skill2: D/E tested (F%)

## Next Focus
- What to explore next

## Self-Assessment
- Should continue: YES/NO
- Reason: ...
```

---

## Exploration Guidelines

### What to Explore
1. **Read ALL skill documentation**: SKILL.md, reference.md for EACH skill
2. **Test each function in each skill**: Basic usage, edge cases, error handling
3. **Discover cross-skill workflows**: Combinations of functions from different skills
4. **Document findings per skill**: Keep detailed notes in checkpoints

### Termination Conditions

Set `exploration_complete: true` when ANY of these is met:
1. **Coverage Complete**: You have tested ≥{coverage_threshold}% of documented functions across ALL skills
2. **Convergence**: 3 consecutive checkpoints with no new discoveries in any skill
3. **Workflow Saturation**: ≥3 meaningful workflows (including cross-skill ones) identified and tested
4. **Explicit Completion**: You genuinely believe ALL skills are fully understood

When setting complete, provide a clear `completion_reason`.

### DO NOT
- Write `exploration_summary.md` until exploration is complete
- Repeat tests that are already in `tested_functions`
- Skip the checkpoint protocol
- Set `exploration_complete: true` prematurely
- Focus only on one skill while ignoring others

### DO
- Update state file after every ~{checkpoint_interval} tool calls
- Build on previous checkpoints (read them if resuming)
- Test edge cases and error conditions
- Document concrete examples, not just descriptions
- Explore ALL skills, not just one

---

## Final Output

When `exploration_complete: true`, write `{output_dir}/exploration_summary.md`:
- Comprehensive summary of ALL skills explored
- All discovered functions per skill and their usage
- Verified workflows (including cross-skill workflows) with examples
- Known limitations and edge cases per skill

This summary will be used for subsequent task generation.

---

Now begin (or continue) your systematic multi-skill exploration.
""".strip()


# 回退摘要生成提示（当 Claude 未自行生成时使用）
FALLBACK_SUMMARY_GENERATION_PROMPT = """Based on the exploration state and checkpoint files, generate a comprehensive exploration summary.

## Exploration State
```json
{exploration_state_json}
```

## Checkpoint Files Content
{checkpoint_files_content}

---

## Task

Generate a comprehensive `exploration_summary.md` that includes:

1. **Skills Explored**: List all skills that were explored
2. **Discovered Functions**: For each skill, list all functions discovered and tested, with brief descriptions
3. **Verified Workflows**: Cross-skill workflows identified during exploration, with concrete examples
4. **Coverage Summary**:
   - Total documented functions vs tested functions
   - Coverage percentage per skill
5. **Known Limitations**: Edge cases, error conditions, or limitations discovered
6. **Key Findings**: Most important discoveries and patterns

Format the output as a complete markdown document that can be saved directly as `exploration_summary.md`.
Do not include any preamble or explanation - output only the markdown content.
""".strip()


# Domain suggestions based on file types
FILE_TYPE_DOMAINS = {
    "pdf": "document analysis",
    "doc": "document processing",
    "docx": "document processing",
    "xlsx": "data analysis",
    "xls": "data analysis",
    "csv": "data analysis",
    "json": "data processing",
    "xml": "data processing",
    "py": "software engineering",
    "js": "software engineering",
    "ts": "software engineering",
    "java": "software engineering",
    "sql": "database",
    "md": "technical writing",
    "stl": "3D modeling",
}


class PromptBuilder:
    """Builder for Claude Code exploration prompts"""

    def __init__(self, config: PromptConfig | None = None):
        """
        Initialize prompt builder.

        Args:
            config: Prompt configuration, uses defaults if None
        """
        self.config = config or PromptConfig()

    def build_exploration_prompt(
        self,
        file_path: str | None,
        skills_hint: str | None = None,
        domain: str | None = None,
        max_steps: int | None = None,
        output_dir: str = "/root/output",
    ) -> str:
        """
        Build an exploration prompt for Claude Code.

        Args:
            file_path: Input file path (None for no-file scenarios)
            skills_hint: Optional hint about available skills
            domain: Optional domain hint (e.g., "data analysis")
            max_steps: Maximum exploration steps (overrides config)
            output_dir: Output directory path

        Returns:
            Formatted prompt string
        """
        max_steps = max_steps or self.config.max_steps
        style = self.config.template_style

        # Use custom template if provided
        if self.config.custom_template:
            return self._format_custom_template(
                file_path, skills_hint, domain, max_steps, output_dir
            )

        # Select appropriate template based on file presence and style
        if file_path:
            template = PROMPT_TEMPLATES.get(style, PROMPT_TEMPLATES[PromptStyle.MINIMAL])
        else:
            template = NO_FILE_TEMPLATES.get(style, NO_FILE_TEMPLATES[PromptStyle.MINIMAL])

        # Auto-detect domain if not provided and config allows
        if self.config.include_domain and not domain and file_path:
            domain = self._detect_domain(file_path)

        # Build the prompt
        return self._format_template(
            template, file_path, skills_hint, domain, max_steps, output_dir
        )

    def _format_template(
        self,
        template: str,
        file_path: str | None,
        skills_hint: str | None,
        domain: str | None,
        max_steps: int,
        output_dir: str,
    ) -> str:
        """Format a template with provided values"""
        # Prepare format kwargs
        format_kwargs = {
            "max_steps": max_steps,
            "output_dir": output_dir,
        }

        if file_path:
            format_kwargs["file_path"] = file_path

        if domain:
            format_kwargs["domain"] = domain
        elif "{domain}" in template:
            format_kwargs["domain"] = "general"

        if skills_hint:
            format_kwargs["skills_hint"] = skills_hint
        elif "{skills_hint}" in template:
            format_kwargs["skills_hint"] = "Use your available tools"

        return template.format(**format_kwargs)

    def _format_custom_template(
        self,
        file_path: str | None,
        skills_hint: str | None,
        domain: str | None,
        max_steps: int,
        output_dir: str,
    ) -> str:
        """Format custom template with available values"""
        template = self.config.custom_template

        replacements = {
            "{file_path}": file_path or "",
            "{skills_hint}": skills_hint or "",
            "{domain}": domain or "",
            "{max_steps}": str(max_steps),
            "{output_dir}": output_dir,
        }

        result = template
        for key, value in replacements.items():
            result = result.replace(key, value)

        return result

    def _detect_domain(self, file_path: str) -> str:
        """Auto-detect domain based on file extension"""
        ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
        return FILE_TYPE_DOMAINS.get(ext, "general")

    def build_summary_prompt(
        self,
        file_path: str,
        max_chars: int = 50,
        max_read_lines: int = 100,
    ) -> str:
        """
        Build a prompt for generating file summary.

        Args:
            file_path: Path to the file
            max_chars: Maximum characters for the summary
            max_read_lines: Maximum lines to read from the file

        Returns:
            Formatted prompt string for summary generation
        """
        return FILE_SUMMARY_PROMPT.format(
            file_path=file_path,
            max_chars=max_chars,
            max_read_lines=max_read_lines,
        )

    def build_exploration_prompt_with_summaries(
        self,
        file_summaries: dict[str, str],
        skills_hint: str | None = None,
        domain: str | None = None,
        max_steps: int | None = None,
        output_dir: str = "/root/output",
        summary_exploration_history: str = "",
    ) -> str:
        """
        Build an exploration prompt with pre-generated file summaries.

        Args:
            file_summaries: Mapping of file paths to their summaries
            skills_hint: Optional hint about available skills
            domain: Optional domain hint (e.g., "data analysis")
            max_steps: Maximum exploration steps (overrides config)
            output_dir: Output directory path

        Returns:
            Formatted prompt string with embedded file summaries
        """
        max_steps = max_steps or self.config.max_steps
        style = self.config.template_style

        # Format file summaries section
        summaries_text = self._format_file_summaries(file_summaries)

        # Use custom template if provided
        if self.config.custom_template:
            return self._format_custom_template_with_summaries(
                summaries_text, skills_hint, domain, max_steps, output_dir
            )

        # Select appropriate template
        template = PROMPT_TEMPLATES_WITH_SUMMARIES.get(
            style, PROMPT_TEMPLATES_WITH_SUMMARIES[PromptStyle.MINIMAL]
        )

        # Auto-detect domain if not provided and config allows
        if self.config.include_domain and not domain and file_summaries:
            # Use first file for domain detection
            first_file = next(iter(file_summaries.keys()))
            domain = self._detect_domain(first_file)

        # Build the prompt
        return self._format_template_with_summaries(
            template, summaries_text, skills_hint, domain, max_steps, output_dir, summary_exploration_history
        )

    def _format_file_summaries(self, file_summaries: dict[str, str]) -> str:
        """Format file summaries into a readable section"""
        if not file_summaries:
            return "[No input files]"

        lines = []
        for file_path, summary in file_summaries.items():
            lines.append(f"File: {file_path}")
            lines.append(f"Summary: {summary}")
            lines.append("")  # Empty line between files

        return "\n".join(lines).strip()

    def _format_template_with_summaries(
        self,
        template: str,
        summaries_text: str,
        skills_hint: str | None,
        domain: str | None,
        max_steps: int,
        output_dir: str,
        summary_exploration_history: str = "",
    ) -> str:
        """Format a template with file summaries"""
        format_kwargs = {
            "max_steps": max_steps,
            "output_dir": output_dir,
            "file_summaries": summaries_text,
            "summary_exploration_history": summary_exploration_history or "[This is the first exploration loop]",
        }

        if domain:
            format_kwargs["domain"] = domain
        elif "{domain}" in template:
            format_kwargs["domain"] = "general"

        if skills_hint:
            format_kwargs["skills_hint"] = skills_hint
        elif "{skills_hint}" in template:
            format_kwargs["skills_hint"] = "Use your available tools"

        return template.format(**format_kwargs)

    def _format_custom_template_with_summaries(
        self,
        summaries_text: str,
        skills_hint: str | None,
        domain: str | None,
        max_steps: int,
        output_dir: str,
    ) -> str:
        """Format custom template with file summaries"""
        template = self.config.custom_template

        replacements = {
            "{file_summaries}": summaries_text,
            "{file_path}": summaries_text,  # Backward compatibility
            "{skills_hint}": skills_hint or "",
            "{domain}": domain or "",
            "{max_steps}": str(max_steps),
            "{output_dir}": output_dir,
        }

        result = template
        for key, value in replacements.items():
            result = result.replace(key, value)

        return result

    def build_batch_prompt(
        self,
        file_paths: list[str],
        output_dir: str = "/root/output",
        max_steps: int | None = None,
    ) -> str:
        """
        Build prompt for processing multiple files.

        Args:
            file_paths: List of input file paths
            output_dir: Output directory path
            max_steps: Maximum steps per file

        Returns:
            Formatted prompt string
        """
        max_steps = max_steps or self.config.max_steps

        files_list = "\n".join(f"- {fp}" for fp in file_paths)

        return f"""
Please process the following files:
{files_list}

For each file, perform appropriate analysis or transformation.
Output all results to {output_dir}.
Total maximum steps: {max_steps * len(file_paths)}.
""".strip()

    @classmethod
    def create_minimal(cls, max_steps: int = 10) -> "PromptBuilder":
        """Create a minimal-style prompt builder"""
        config = PromptConfig(
            template_style=PromptStyle.MINIMAL,
            max_steps=max_steps,
        )
        return cls(config)

    @classmethod
    def create_domain_guided(cls, max_steps: int = 10) -> "PromptBuilder":
        """Create a domain-guided prompt builder"""
        config = PromptConfig(
            template_style=PromptStyle.DOMAIN_GUIDED,
            max_steps=max_steps,
            include_domain=True,
        )
        return cls(config)

    @classmethod
    def create_skill_hinted(cls, max_steps: int = 10) -> "PromptBuilder":
        """Create a skill-hinted prompt builder"""
        config = PromptConfig(
            template_style=PromptStyle.SKILL_HINTED,
            max_steps=max_steps,
            include_skills_hint=True,
        )
        return cls(config)

    @classmethod
    def create_goal_oriented(cls, max_steps: int = 10) -> "PromptBuilder":
        """Create a goal-oriented prompt builder"""
        config = PromptConfig(
            template_style=PromptStyle.GOAL_ORIENTED,
            max_steps=max_steps,
        )
        return cls(config)

    def build_exploration_prompt_with_history(
        self,
        file_summaries: dict[str, str],
        previous_summary: str,
        skills_hint: str | None = None,
        max_steps: int | None = None,
        output_dir: str = "/root/output",
        exploration_goals: str | None = None,
    ) -> str:
        """
        Build an exploration prompt that includes history from previous loops.

        Args:
            file_summaries: Mapping of file paths to their summaries
            previous_summary: Summary text from previous exploration loops
            skills_hint: Optional hint about available skills
            max_steps: Maximum exploration steps (overrides config)
            output_dir: Output directory path
            exploration_goals: Optional specific goals for this exploration loop

        Returns:
            Formatted prompt string with embedded history
        """
        max_steps = max_steps or self.config.max_steps

        # Format file summaries section
        summaries_text = self._format_file_summaries(file_summaries)

        # Default exploration goals if not provided
        if not exploration_goals:
            exploration_goals = """
1. Explore skill capabilities not yet discovered
2. Test different parameter combinations
3. Find new workflows and patterns
4. Produce verifiable outputs
""".strip()

        return EXPLORATION_WITH_HISTORY_TEMPLATE.format(
            previous_summary=previous_summary,
            skills_hint=skills_hint or "Use your available tools",
            file_summaries=summaries_text,
            output_dir=output_dir,
            max_steps=max_steps,
            exploration_goals=exploration_goals,
        )

    def build_goal_driven_prompt(
        self,
        task_instruction: str,
        execution_guide: str,
        skills_hint: str,
        file_summaries: dict[str, str],
        output_dir: str = "/root/output",
    ) -> str:
        """
        Build a goal-driven execution prompt.

        与 PROMPT_TEMPLATES_WITH_SUMMARIES[PromptStyle.MINIMAL] 保持一致的参数命名。

        Args:
            task_instruction: 任务指令（简洁描述，来自 Phase 1.5）
            execution_guide: 执行指南/元信息（详细的做题思路，来自 Phase 1.5）
            skills_hint: 可用技能列表（与探索模板一致）
            file_summaries: 文件路径到摘要的映射（与探索模板一致）
            output_dir: 输出目录路径（与 exploration_output_dir 一致）

        Returns:
            Formatted prompt string for goal-driven execution
        """
        # Format file summaries section (与探索模板使用相同的格式化方法)
        summaries_text = self._format_file_summaries(file_summaries)

        return GOAL_DRIVEN_TEMPLATE.format(
            task_instruction=task_instruction,
            execution_guide=execution_guide or "[No additional guidance]",
            skills_hint=skills_hint or "Use your available skills",
            file_summaries=summaries_text,
            output_dir=output_dir,
        )

    def build_checkpoint_exploration_prompt(
        self,
        file_summaries: dict[str, str],
        skills_hint: str | None,
        skill_names: list[str],
        documented_functions: dict[str, list[str]],
        current_state_json: str,
        output_dir: str,
        chunk_index: int,
        checkpoint_interval: int = 20,
        previous_summary: str | None = None,
        coverage_threshold: float = 0.9,
    ) -> str:
        """
        Build a checkpoint-based exploration prompt for multiple skills.

        Args:
            file_summaries: Mapping of file paths to their summaries
            skills_hint: Skills directory hint
            skill_names: List of skill names being explored
            documented_functions: Mapping of skill name to function lists
            current_state_json: JSON string of current ExplorationState
            output_dir: Output directory path
            chunk_index: Current chunk index (1-based)
            checkpoint_interval: Number of tool calls between checkpoints (default: 20)
            previous_summary: Content of previous exploration summary to merge (optional)

        Returns:
            Formatted prompt string for checkpoint-based exploration
        """
        import json

        # Format file summaries
        summaries_text = self._format_file_summaries(file_summaries)

        # Format skill names for display
        skill_names_str = ", ".join(skill_names) if skill_names else "unknown"

        # Build resume instruction if continuing
        resume_instruction = ""
        coverage_pct_display = int(coverage_threshold * 100)
        if chunk_index > 1:
            resume_instruction = f"""
## ⚠️ RESUMING EXPLORATION (Chunk #{chunk_index})

Your prior work is preserved in:
- `{output_dir}/exploration_state.json` - Current progress state
- `{output_dir}/checkpoint_*.md` - Previous detailed notes

**CRITICAL - Coverage Target is {coverage_pct_display}%**:

The exploration is NOT complete until coverage reaches {coverage_pct_display}%. Your current coverage is below this target, which is why exploration is resuming.

**MUST DO**:
1. Read the state file FIRST to understand what's already done
2. Identify functions NOT in `tested_functions` - those are your targets
3. **When you test a new function, IMMEDIATELY add it to `tested_functions` in the state file**
4. Do NOT set `exploration_complete: true` until coverage ≥ {coverage_pct_display}%

**MUST NOT**:
- Repeat tests for functions already in `tested_functions`
- Mark exploration complete before reaching {coverage_pct_display}% coverage
- Forget to update `tested_functions` after testing a function
"""
            # Add previous summary content for merging
            if previous_summary:
                resume_instruction += f"""
## 📋 PREVIOUS EXPLORATION SUMMARY (from earlier chunks)

The following summary was generated in a previous exploration chunk. When you generate the final `exploration_summary.md`, you MUST:
1. **MERGE** this previous summary with your new findings
2. **DO NOT discard** any information from the previous summary
3. **UPDATE** sections with new discoveries and tested functions
4. **KEEP** all previously documented workflows, functions, and notes

<previous_summary>
{previous_summary[:8000]}{"... [truncated]" if len(previous_summary) > 8000 else ""}
</previous_summary>
"""

        return CHECKPOINT_EXPLORATION_PROMPT.format(
            resume_instruction=resume_instruction,
            current_state_json=current_state_json,
            documented_functions_json=json.dumps(documented_functions, indent=2, ensure_ascii=False),
            skills_hint=skills_hint or "Use your available skills",
            file_summaries=summaries_text,
            output_dir=output_dir,
            skill_names=skill_names_str,
            checkpoint_count=chunk_index,
            checkpoint_interval=checkpoint_interval,
            coverage_threshold=coverage_pct_display,
        )

    def build_dag_exploration_prompt(
        self,
        dag_task: "DAGTask",
        dag_state: "DAGExplorationState",
        file_summaries: dict[str, str],
        skills_docs: dict[str, str],
        output_dir: str,
        chunk_index: int,
        checkpoint_interval: int = 20,
        previous_summary: str | None = None,
    ) -> str:
        """
        Build a DAG-guided exploration prompt.

        Args:
            dag_task: DAGTask object describing the skill DAG
            dag_state: Current DAGExplorationState
            file_summaries: file path -> summary
            skills_docs: skill name -> SKILL.md content (or path)
            output_dir: Output directory
            chunk_index: Current chunk (1-based)
            checkpoint_interval: Tool calls between checkpoints
            previous_summary: Previous exploration summary to merge

        Returns:
            Formatted prompt string
        """
        import json
        from ..prompts.dag_exploration import (
            DAG_EXPLORATION_PROMPT,
            DAG_EXPLORATION_RESUME,
        )

        # Build nodes list
        nodes_list = "\n".join(
            f"- **{s['skill_name']}** (id: {s.get('skill_id', '?')})"
            for s in dag_task.skills
        )

        # Build edges list with scenario descriptions
        edges_lines = []
        for e in dag_task.edges:
            desc = f" — {e.scenario_description[:120]}" if e.scenario_description else ""
            edges_lines.append(f"- {e.source_skill} -> {e.target_skill} [{', '.join(e.alignment_types)}]{desc}")
        edges_list = "\n".join(edges_lines)

        # Topological order
        topo = dag_task.topological_order()
        topological_order = " -> ".join(topo)

        # All paths display
        paths = dag_task.all_paths()
        all_paths_display = "\n".join(
            f"{i+1}. {' -> '.join(p)}" for i, p in enumerate(paths)
        )

        # Suggested workflow
        suggested = dag_task.suggested_task
        if suggested:
            workflow_steps = suggested.get("workflow", [])
            if isinstance(workflow_steps, list):
                suggested_workflow = "\n".join(
                    f"  {i+1}. {step}" if isinstance(step, str) else f"  {i+1}. {step.get('description', str(step))}"
                    for i, step in enumerate(workflow_steps)
                )
            else:
                suggested_workflow = str(workflow_steps)
        else:
            suggested_workflow = "[No suggested workflow]"

        # Skills docs
        docs_lines = []
        for skill_name, doc_content in skills_docs.items():
            if len(doc_content) > 3000:
                doc_content = doc_content[:3000] + "\n... [truncated]"
            docs_lines.append(f"### {skill_name}\n{doc_content}")
        skills_docs_text = "\n\n".join(docs_lines) if docs_lines else "[No documentation]"

        # File summaries
        summaries_text = self._format_file_summaries(file_summaries)

        # State JSON
        current_state_json = json.dumps(dag_state.to_dict(), indent=2, ensure_ascii=False)

        # Resume instruction
        resume_instruction = ""
        if chunk_index > 1:
            resume_instruction = DAG_EXPLORATION_RESUME.format(
                chunk_index=chunk_index,
                output_dir=output_dir,
            )
            if previous_summary:
                resume_instruction += f"""
## Previous Exploration Summary (from earlier chunks)

When you generate the final `exploration_summary.md`, MERGE this with your new findings:

<previous_summary>
{previous_summary[:8000]}{"... [truncated]" if len(previous_summary) > 8000 else ""}
</previous_summary>
"""

        return DAG_EXPLORATION_PROMPT.format(
            structure_type=dag_task.structure_type,
            nodes_list=nodes_list,
            edges_list=edges_list,
            topological_order=topological_order,
            all_paths_display=all_paths_display,
            suggested_workflow=suggested_workflow,
            current_state_json=current_state_json,
            skills_docs=skills_docs_text,
            file_summaries=summaries_text,
            output_dir=output_dir,
            checkpoint_interval=checkpoint_interval,
            dag_task_id=dag_task.task_id,
            skill_names_json=json.dumps(dag_task.skill_names),
            all_paths_json=json.dumps(paths),
            all_edges_json=json.dumps([f"{e.source_skill}->{e.target_skill}" for e in dag_task.edges]),
            checkpoint_count=chunk_index,
            resume_instruction=resume_instruction,
        )
