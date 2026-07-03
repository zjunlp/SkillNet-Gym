"""File summarizer module for generating summaries and classifying content types"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..config import ContentType, FileSummaryEntry, FileSummaryResult
from ..utils.llm_client import LLMClient


# Prompt for content type classification
CONTENT_TYPE_CLASSIFICATION_PROMPT = """Based on the file summary below, classify the file into one of these content types:

- **form**: Files with fillable fields, input boxes, checkboxes, dropdown menus (e.g., tax forms, application forms, surveys)
- **text**: Pure text documents, articles, reports, contracts without interactive elements
- **table**: Files primarily containing tabular data (spreadsheets, CSV-like data, data tables)
- **code**: Source code files, scripts, configuration files
- **mixed**: Files combining multiple content types

## File Summary
{summary}

## File Name
{filename}

---

Respond with ONLY the content type (one word): form, text, table, code, or mixed
"""


class FileSummarizer:
    """Generates summaries for files in a folder and classifies content types"""

    def __init__(
        self,
        executor: Any = None,  # ClaudeExecutor for file summary generation
        llm_client: LLMClient | None = None,  # For content type classification
        max_workers: int = 3,  # 并发数
        show_progress: bool = True,  # 是否显示进度条
        extract_metadata: bool = False,  # 是否提取详细 metadata
    ):
        """
        Initialize the file summarizer.

        Args:
            executor: ClaudeExecutor instance for generating file summaries
            llm_client: LLM client for content type classification
            max_workers: Maximum number of concurrent workers
            show_progress: Whether to show progress bar
            extract_metadata: Whether to extract detailed metadata for each file
        """
        self.executor = executor
        self.llm = llm_client
        self.max_workers = max_workers
        self.show_progress = show_progress
        self.extract_metadata = extract_metadata
        self._metadata_extractor = None  # Lazy initialization

    @property
    def metadata_extractor(self):
        """Lazy initialization of MetadataExtractor."""
        if self._metadata_extractor is None and self.extract_metadata:
            from .metadata_extractor import MetadataExtractor
            self._metadata_extractor = MetadataExtractor(llm_client=self.llm)
        return self._metadata_extractor

    def summarize_folder(
        self,
        folder_path: str,
        output_json: str | None = None,
        file_extensions: list[str] | None = None,
        max_files: int | None = None,
        ignore_files: list[str] | None = None,
    ) -> FileSummaryResult:
        """
        Generate summaries for all files in a folder.

        Args:
            folder_path: Path to the folder containing files
            output_json: Path to save the result JSON (optional)
            file_extensions: List of file extensions to include (e.g., ['.pdf', '.xlsx'])
                           If None, includes all files
            max_files: Maximum number of files to process (optional)
            ignore_files: List of file names to ignore (e.g., ['requirements.txt', 'download_report.json'])

        Returns:
            FileSummaryResult with all file summaries and content type groupings
        """
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            raise ValueError(f"Invalid folder path: {folder_path}")

        # Default ignore list
        ignore_set = set(ignore_files or [])

        # Collect files
        files_to_process = []
        for item in folder.iterdir():
            if item.is_file():
                # Skip ignored files
                if item.name in ignore_set:
                    continue
                if file_extensions is None or item.suffix.lower() in file_extensions:
                    files_to_process.append(item)

        # Sort by name for consistency
        files_to_process.sort(key=lambda x: x.name)

        # Apply max limit
        if max_files is not None:
            files_to_process = files_to_process[:max_files]

        print(f"[FileSummarizer] Found {len(files_to_process)} files to process (workers={self.max_workers})")

        # Process files concurrently
        entries: list[FileSummaryEntry] = []
        content_types: dict[str, list[str]] = {}

        def process_single_file(file_path: Path) -> FileSummaryEntry:
            """Process a single file and return entry"""
            try:
                return self.summarize_file(str(file_path))
            except Exception as e:
                return FileSummaryEntry(
                    name=file_path.name,
                    path=str(file_path),
                    summary=f"[Error: {str(e)[:100]}]",
                    content_type=ContentType.MIXED.value,
                    metadata={"error": str(e)},
                )

        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(process_single_file, fp): fp
                for fp in files_to_process
            }

            # Process results with progress bar
            pbar = tqdm(
                as_completed(future_to_path),
                total=len(files_to_process),
                desc="Summarizing files",
                disable=not self.show_progress,
            )

            for future in pbar:
                file_path = future_to_path[future]
                try:
                    entry = future.result()
                    entries.append(entry)

                    # Group by content type
                    ct = entry.content_type
                    if ct not in content_types:
                        content_types[ct] = []
                    content_types[ct].append(entry.path)

                    # Print summary preview (first 30 chars)
                    summary_preview = entry.summary[:100].replace("\n", " ")
                    tqdm.write(f"  [{ct}] {entry.name}: {summary_preview}...")

                    # Update progress bar description
                    pbar.set_postfix_str(f"{entry.name} -> {ct}")

                except Exception as e:
                    print(f"[FileSummarizer] Error processing {file_path.name}: {e}")
                    entries.append(FileSummaryEntry(
                        name=file_path.name,
                        path=str(file_path),
                        summary=f"[Error: {str(e)[:100]}]",
                        content_type=ContentType.MIXED.value,
                        metadata={"error": str(e)},
                    ))

        # Sort entries by name for consistent ordering
        entries.sort(key=lambda x: x.name)

        result = FileSummaryResult(
            files=entries,
            content_types=content_types,
        )

        # Save to JSON if path provided
        if output_json:
            self.save_to_json(result, output_json)
            print(f"[FileSummarizer] Saved to {output_json}")

        return result

    def summarize_file(
        self,
        file_path: str,
        max_chars: int = 200,
        max_read_lines: int = 500,
    ) -> FileSummaryEntry:
        """
        Generate summary for a single file and classify its content type.

        Args:
            file_path: Path to the file
            max_chars: Maximum characters for the summary
            max_read_lines: Maximum lines to read from large files

        Returns:
            FileSummaryEntry with summary and content type
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Generate summary using executor
        if self.executor is not None:
            summary = self.executor.generate_file_summary(
                file_path=str(path),
                max_chars=max_chars,
                max_read_lines=max_read_lines,
            )
        else:
            # Fallback: read first part of file
            summary = self._generate_basic_summary(str(path), max_read_lines)

        # Classify content type
        content_type = self.classify_content_type(summary, path.name)

        # Extract detailed metadata if enabled
        metadata = {}
        if self.extract_metadata and self.metadata_extractor:
            try:
                metadata = self.metadata_extractor.extract(
                    file_path=str(path),
                    summary=summary,
                    content_type=content_type,
                )
            except Exception as e:
                metadata = {"extraction_error": str(e)[:200]}

        return FileSummaryEntry(
            name=path.name,
            path=str(path.resolve()),
            summary=summary,
            content_type=content_type,
            metadata=metadata,
        )

    def classify_content_type(self, summary: str, filename: str) -> str:
        """
        Classify file content type based on summary.

        Uses LLM if available, otherwise uses heuristics.

        Args:
            summary: File content summary
            filename: File name for additional context

        Returns:
            Content type string (form, text, table, code, mixed)
        """
        if self.llm is not None:
            return self._classify_with_llm(summary, filename)
        else:
            return self._classify_with_heuristics(summary, filename)

    def _classify_with_llm(self, summary: str, filename: str) -> str:
        """Classify using LLM"""
        prompt = CONTENT_TYPE_CLASSIFICATION_PROMPT.format(
            summary=summary,
            filename=filename,
        )

        try:
            response = self.llm.generate(
                system_prompt="You are a file content classifier. Respond with only one word.",
                user_prompt=prompt,
                temperature=0.1,
            )

            # Parse response
            response = response.strip().lower()
            valid_types = [ct.value for ct in ContentType]

            if response in valid_types:
                return response
            else:
                # Try to extract type from response
                for ct in valid_types:
                    if ct in response:
                        return ct
                return ContentType.MIXED.value

        except Exception as e:
            print(f"[FileSummarizer] LLM classification failed: {e}, using heuristics")
            return self._classify_with_heuristics(summary, filename)

    def _classify_with_heuristics(self, summary: str, filename: str) -> str:
        """Classify using keyword heuristics"""
        summary_lower = summary.lower()
        filename_lower = filename.lower()

        # Form indicators
        form_keywords = [
            "form", "fillable", "field", "input", "checkbox", "dropdown",
            "signature", "w-4", "w-9", "application", "submit", "填写",
        ]
        if any(kw in summary_lower for kw in form_keywords):
            return ContentType.FORM.value

        # Table indicators
        table_keywords = [
            "table", "row", "column", "spreadsheet", "xlsx", "csv",
            "data", "cell", "header", "表格", "数据",
        ]
        if any(kw in summary_lower for kw in table_keywords):
            return ContentType.TABLE.value

        # Code indicators
        code_keywords = [
            "function", "class", "import", "def ", "const ", "var ",
            "code", "script", "programming", "syntax",
        ]
        code_extensions = [".py", ".js", ".ts", ".java", ".cpp", ".go", ".rs"]
        if any(kw in summary_lower for kw in code_keywords):
            return ContentType.CODE.value
        if any(filename_lower.endswith(ext) for ext in code_extensions):
            return ContentType.CODE.value

        # Default to text
        return ContentType.TEXT.value

    def _generate_basic_summary(self, file_path: str, max_lines: int = 100) -> str:
        """Generate basic summary by reading file content"""
        path = Path(file_path)

        # For binary files, just return file info
        binary_extensions = [".pdf", ".xlsx", ".xls", ".doc", ".docx", ".png", ".jpg"]
        if path.suffix.lower() in binary_extensions:
            return f"Binary file: {path.suffix} format, size {path.stat().st_size} bytes"

        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line)
                content = "".join(lines)
                return content[:500] if len(content) > 500 else content
        except Exception as e:
            return f"[Unable to read file: {e}]"

    def save_to_json(self, result: FileSummaryResult, output_path: str) -> None:
        """
        Save FileSummaryResult to JSON file.

        Args:
            result: FileSummaryResult to save
            output_path: Path to output JSON file
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    def load_from_json(self, json_path: str) -> FileSummaryResult:
        """
        Load FileSummaryResult from JSON file.

        Args:
            json_path: Path to JSON file

        Returns:
            FileSummaryResult loaded from file
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return FileSummaryResult.from_dict(data)
