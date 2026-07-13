"""
loaders.py - File loading for different file types.

Tabular files (csv, xlsx, tsv, json-array)  -> pandas DataFrame  (shown as grid)
Text / code / pdf files                     -> plain text        (chat Q&A / fix)
"""

from __future__ import annotations
import io
import json
import pandas as pd


TABULAR_EXT = {".csv", ".tsv", ".xlsx", ".xls"}
CODE_EXT = {
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".cs", ".go", ".rb",
    ".php", ".rs", ".swift", ".kt", ".sql", ".sh", ".html", ".css",
    ".jsx", ".tsx", ".r", ".m", ".scala", ".pl",
}
TEXT_EXT = {".txt", ".md", ".log", ".cfg", ".ini", ".yaml", ".yml", ".toml", ".env"}


def file_ext(name: str) -> str:
    """Return the lowercase extension including the leading dot, or ''."""
    return ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""


def load_file(name: str, data: bytes):
    """
    Load a file's bytes into a normalized dict.

    Returns:
        {
            "kind": "table" | "text" | "code",
            "df": DataFrame | None,
            "text": str | None,
            "ext": str,
            "name": str,
        }
    """
    ext = file_ext(name)

    # ---- Tabular ----
    if ext == ".csv":
        df = _read_csv(data, sep=",")
        return _table(name, ext, df)
    if ext == ".tsv":
        df = _read_csv(data, sep="\t")
        return _table(name, ext, df)
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(io.BytesIO(data))
        return _table(name, ext, df)
    if ext == ".json":
        df = _try_json_table(data)
        if df is not None:
            return _table(name, ext, df)
        # Fall back to treating it as text/code
        return _text(name, ".json", data.decode("utf-8", errors="replace"), kind="code")

    # ---- PDF ----
    if ext == ".pdf":
        return _text(name, ext, _read_pdf(data), kind="text")

    # ---- Code ----
    if ext in CODE_EXT:
        return _text(name, ext, data.decode("utf-8", errors="replace"), kind="code")

    # ---- Plain text (default) ----
    return _text(name, ext or ".txt", data.decode("utf-8", errors="replace"), kind="text")


def _read_csv(data: bytes, sep=","):
    """Read a CSV with simple encoding auto-detection."""
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), sep=sep, encoding=enc)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    # Last resort: python engine, skip malformed lines
    return pd.read_csv(io.BytesIO(data), sep=sep, encoding="latin-1",
                       engine="python", on_bad_lines="skip")


def _try_json_table(data: bytes):
    """Try to parse JSON as a table. Returns a DataFrame or None."""
    try:
        obj = json.loads(data.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return pd.json_normalize(obj)
    if isinstance(obj, dict):
        # Dict of lists -> DataFrame
        if all(isinstance(v, list) for v in obj.values()) and obj:
            try:
                return pd.DataFrame(obj)
            except ValueError:
                return None
    return None


def _read_pdf(data: bytes) -> str:
    """Extract text from a PDF using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return "[pypdf is required to read PDFs. Install with:  pip install pypdf]"
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages, 1):
        txt = page.extract_text() or ""
        pages.append(f"--- Page {i} ---\n{txt}")
    return "\n\n".join(pages).strip() or (
        "[No extractable text found in this PDF - it may be a scanned image.]"
    )


def _table(name, ext, df):
    return {"kind": "table", "df": df, "text": None, "ext": ext, "name": name}


def _text(name, ext, text, kind="text"):
    return {"kind": kind, "df": None, "text": text, "ext": ext, "name": name}
