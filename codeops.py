"""
codeops.py - Code file operations: fix errors, add/remove lines, explain,
and show a unified diff of the change.

Strategy: send the LLM the full file plus the user's instruction and ask for
the FULL updated file back inside a fenced code block. We extract only the
code block, then compute an old-vs-new unified diff so the user can see
exactly what changed.
"""

from __future__ import annotations
import difflib


FIX_SYSTEM = """You are an expert programmer and code reviewer.
The user will give you a source file and an instruction (fix errors, add a feature,
remove/add lines, refactor, explain, etc.).

When the instruction requires CHANGING the code:
1. First give a short explanation of what was wrong / what you changed (3-6 bullet lines max).
2. Then output the COMPLETE corrected file inside a single fenced code block with the
   correct language tag, e.g. ```python ... ```.
   - Output the ENTIRE file, not just the changed parts.
   - Preserve the original style and indentation where possible.

When the instruction is only a QUESTION (no change needed), just answer concisely
and do NOT output a code block.

Be precise. Do not invent APIs. If something is ambiguous, make the most reasonable choice
and note it briefly.
"""


def lang_for_ext(ext: str) -> str:
    """Map a file extension to a Markdown code-fence language tag."""
    return {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".jsx": "jsx", ".tsx": "tsx", ".java": "java", ".c": "c",
        ".cpp": "cpp", ".cs": "csharp", ".go": "go", ".rb": "ruby",
        ".php": "php", ".rs": "rust", ".swift": "swift", ".kt": "kotlin",
        ".sql": "sql", ".sh": "bash", ".html": "html", ".css": "css",
        ".r": "r", ".scala": "scala", ".pl": "perl", ".json": "json",
    }.get(ext, "")


def ask_code(llm, code: str, ext: str, instruction: str, lang: str = "English"):
    """
    Ask the LLM to apply an instruction to a source file.

    Returns (explanation_text, new_code_or_None). When ``new_code`` is None,
    the LLM only answered a question and did not modify the code.
    """
    lang_sys = FIX_SYSTEM + f"\n\nWrite the explanation / any prose in {lang}."
    code_lang = lang_for_ext(ext) or "text"
    user = (
        f"Instruction: {instruction}\n\n"
        f"Here is the current file ({code_lang}):\n"
        f"```{code_lang}\n{code}\n```"
    )
    reply = llm.ask(lang_sys, user, temperature=0.1)
    new_code = _extract_code_block(reply)
    explanation = _strip_code_block(reply) if new_code else reply
    return explanation.strip(), new_code


def _extract_code_block(text: str):
    """Return the largest fenced code block from a response, or None."""
    if "```" not in text:
        return None
    blocks = []
    parts = text.split("```")
    # parts: [before, block1, between, block2, ...] - odd indices are blocks
    for i in range(1, len(parts), 2):
        block = parts[i]
        # First line may be a language tag; strip it if so
        nl = block.find("\n")
        if nl != -1:
            first = block[:nl].strip()
            if first and " " not in first and len(first) < 20:
                block = block[nl + 1:]
        blocks.append(block.rstrip("\n"))
    if not blocks:
        return None
    return max(blocks, key=len)


def _strip_code_block(text: str) -> str:
    """Return only the prose portions of a response, dropping code blocks."""
    if "```" not in text:
        return text
    parts = text.split("```")
    # Even indices are outside any code fence
    outside = [parts[i] for i in range(0, len(parts), 2)]
    return "\n".join(p.strip() for p in outside if p.strip())


def make_diff(old: str, new: str, filename: str) -> str:
    """Return a unified diff string comparing old vs. new file contents."""
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        lineterm="",
    )
    return "".join(diff)


def line_stats(old: str, new: str):
    """Return (added_lines, removed_lines) between two file contents."""
    sm = difflib.SequenceMatcher(a=old.splitlines(), b=new.splitlines())
    added = removed = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            removed += (i2 - i1)
            added += (j2 - j1)
        elif tag == "delete":
            removed += (i2 - i1)
        elif tag == "insert":
            added += (j2 - j1)
    return added, removed
