"""Prompt for selecting representative files for skill exploration"""

SYS_PROMPT = """You are a file analysis expert. Your task is to select the most representative files for skill exploration from a list of file summaries.

Selection criteria:
1. Each content type should have exactly ONE representative file
2. The selected file must have rich content (NOT empty or minimal information)
3. The file should best demonstrate the typical characteristics of its content type
4. Prefer files with clear structure and meaningful data"""

USER_PROMPT = """# File Summaries

Below are summaries of files grouped by content type:

{file_summaries_by_type}

---

# Selection Task

For each content type present, select exactly ONE file that best represents that type for skill exploration.

## Selection Guidelines

1. **Richness**: The file must contain substantial content, NOT be empty or have minimal data
   - Reject files described as "empty", "blank", "no content", "minimal", "placeholder"
   - Reject files with very short summaries indicating lack of content

2. **Representativeness**: The file should demonstrate typical characteristics of its content type
   - Form files should have multiple fillable fields (text inputs, checkboxes, dropdowns)
   - Text files should have structured content (paragraphs, sections, meaningful text)
   - Table files should have multiple rows and columns with actual data
   - Code files should contain actual implementation, not just comments or stubs

3. **Exploration Value**: The file should allow comprehensive skill testing
   - Prefer files that exercise multiple features of the skill
   - Prefer files with moderate complexity (not too simple, not overly complex)

## Content Type Definitions

- **form**: Files with fillable fields, input boxes, checkboxes (e.g., tax forms, application forms, surveys)
- **text**: Pure text documents, articles, reports without interactive elements
- **table**: Files primarily containing tabular data (spreadsheets, CSV-like content)
- **code**: Source code files, scripts, configuration files
- **mixed**: Files combining multiple content types

---

# Output Format

Return a JSON object with your selections:

```json
{{
  "selections": [
    {{
      "content_type": "form",
      "selected_file": "/full/path/to/file.pdf",
      "reason": "This tax form has 15+ fillable fields including text inputs, checkboxes, and dropdown menus, making it ideal for testing form-handling workflows."
    }},
    {{
      "content_type": "text",
      "selected_file": "/full/path/to/document.pdf",
      "reason": "This document contains 10+ pages of structured text with headers, paragraphs, and lists, suitable for comprehensive text extraction testing."
    }}
  ],
  "skipped_types": [
    {{
      "content_type": "table",
      "reason": "All 3 table files in this batch are empty or contain only headers without actual data rows."
    }}
  ]
}}
```

## Important Rules

1. Do NOT select files with these indicators in their summary:
   - "empty", "blank", "no content", "no data"
   - "placeholder", "template only", "sample"
   - Very brief summaries (< 20 words) without meaningful content description

2. If ALL files of a content type lack sufficient content, add that type to `skipped_types` with explanation

3. It's acceptable to have fewer selections if some content types have no quality files

4. Always use the full file path as shown in the summaries

5. Provide clear reasons that reference specific features mentioned in the file summary
"""
