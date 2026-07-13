"""
dataops.py - Turn natural-language chat into safe DataFrame operations.

The LLM is asked to translate the user's request into a structured JSON plan;
we then apply that plan with pandas. No arbitrary code is ever executed.

Supported actions:
  - search        : find text across some/all columns
  - filter        : apply a column condition (==, !=, >, <, >=, <=, contains, ...)
  - sort          : sort by one or more columns
  - add           : append new rows to the table
  - add_column    : add a computed column from a formula (e.g. total = price * qty)
  - delete_column : drop a column
  - rename_column : rename a column
  - reset         : restore the original data
  - answer        : (not a data op - the LLM just answers a question)
"""

from __future__ import annotations
import re
import pandas as pd


PLAN_SYSTEM = """You convert a user's natural-language request about a table into a JSON plan.
Return ONLY valid JSON, no prose, no markdown fences.

The table columns are: {columns}
Column types: {dtypes}
Sample rows (CSV):
{sample}

Schema:
{{
  "action": "search|filter|sort|add|add_column|delete_column|rename_column|reset|answer",
  "query": "text to search for (search only)",
  "columns": ["columns to search in - empty means all (search only)"],
  "conditions": [
    {{"column": "<col>", "op": "==|!=|>|<|>=|<=|contains|startswith|endswith", "value": <val>}}
  ],
  "sort_by": ["<col>", ...],
  "ascending": [true/false per sort_by column],
  "new_rows": [ {{"<col>": <value>, ...}}, ... ],
  "new_column": "name of the column to create / rename to",
  "formula": "pandas expression using existing column names, e.g. price * quantity",
  "target_column": "existing column to delete or rename",
  "explanation": "one short sentence in {lang} describing what you did"
}}

Rules:
- Use ONLY column names from the list above (case-insensitive match, output the EXACT name).
- ADD (new rows): triggers like "add 10 records" or "add these rows". Put rows in "new_rows",
  one object per row with a value for EVERY column, realistic and matching the sample's types.
  If the user pasted specific values, use those EXACT values.
- ADD_COLUMN (formula / computed column): triggers like "add a column total = price*qty",
  "compute BMI", "add a profit column". Set action "add_column", "new_column" (the name)
  and "formula" (a math expression using existing column names only,
  e.g. "weight / (height ** 2)").
- DELETE_COLUMN: set action "delete_column" and "target_column".
- RENAME_COLUMN: set action "rename_column", "target_column" (old) and "new_column" (new).
- NEVER treat words like "add", "please", "kro", "karo", "jodo" as filter values.
- A question about the data ("how many rows", "average of X") -> action "answer".
- "show rows where price > 100" -> filter. "sort by date newest first" -> sort, ascending [false].
- "find John" -> search, query "John", columns []. Use "reset" only when the user
  explicitly asks to clear filters / show all data.
- Keep numeric values as numbers, not strings. Write "explanation" in {lang}.
"""


def make_plan(llm, user_msg, df, lang="English"):
    """Build a JSON plan for the user's request against the given DataFrame."""
    cols = list(df.columns)
    dtypes = "; ".join(f"{c}: {df[c].dtype}" for c in cols)
    sample = df.head(3).to_csv(index=False)
    sys = PLAN_SYSTEM.format(columns=", ".join(map(str, cols)),
                             dtypes=dtypes, sample=sample, lang=lang)
    plan = llm.ask_json(sys, user_msg)
    if not isinstance(plan, dict):
        return {"action": "answer",
                "explanation": "Could not parse request as a data operation."}
    return plan


def _resolve_col(name, columns):
    """Return the actual column name for a case-insensitive match, or None."""
    if name in columns:
        return name
    low = {str(c).lower(): c for c in columns}
    return low.get(str(name).lower())


# Excel-style functions we intentionally don't support in the formula engine.
# If we see one in either the user's prompt or the LLM-translated formula, we
# stop rather than silently miscompute (which is what happens if the LLM
# "translates" MAX(...) into per-row col*col).
_UNSUPPORTED_FN_PATTERN = re.compile(
    r"(?i)\b("
    r"IFERROR|IFNA|IFS|IF|"
    r"INDEX|MATCH|VLOOKUP|HLOOKUP|XLOOKUP|LOOKUP|OFFSET|INDIRECT|"
    r"MAX|MIN|AVERAGE|AVG|MEDIAN|MODE|STDEV|VAR|SUM|COUNT|COUNTA|COUNTIF|SUMIF|SUMIFS|AVERAGEIF|"
    r"CONCATENATE|TEXTJOIN|LEFT|RIGHT|MID|LEN|TRIM|UPPER|LOWER|PROPER|SUBSTITUTE|"
    r"NOW|TODAY|DATE|YEAR|MONTH|DAY|WEEKDAY|"
    r"ROUND|ROUNDUP|ROUNDDOWN|CEILING|FLOOR|ABS|MOD|POWER|SQRT"
    r")\s*\("
)


def _unsupported_functions(*texts):
    """Return the sorted list of Excel-style function names found in any of the
    given strings, e.g. ['AVERAGE', 'INDEX', 'MATCH', 'MAX']."""
    found = set()
    for t in texts:
        if not t:
            continue
        for m in _UNSUPPORTED_FN_PATTERN.finditer(str(t)):
            found.add(m.group(1).upper())
    return sorted(found)


def _eval_formula(df, formula):
    """
    Safely evaluate a column formula, handling spaces and mixed casing in
    column names. Each column is aliased to a safe identifier (e.g. _col0_)
    and evaluated in a namespace containing only those aliases (no builtins).
    """
    cols = list(df.columns)
    alias = {c: f"_col{i}_" for i, c in enumerate(cols)}
    expr = str(formula).replace("`", "")  # strip backticks
    # Replace the longest column names first to avoid partial-overlap issues;
    # matching is case-insensitive so "col a" matches "col A".
    for c in sorted(cols, key=lambda x: len(str(x)), reverse=True):
        expr = re.compile(re.escape(str(c)), re.IGNORECASE).sub(alias[c], expr)
    namespace = {alias[c]: df[c] for c in cols}
    return eval(expr, {"__builtins__": {}}, namespace)


def apply_plan(df: pd.DataFrame, plan: dict, user_prompt: str = ""):
    """
    Apply a JSON plan to a DataFrame.

    Returns (result_df, message, did_modify_view). The input DataFrame is
    never mutated.
    """
    action = (plan.get("action") or "answer").lower()
    cols = list(df.columns)

    if action == "reset":
        return df, "View reset - showing all data.", True

    if action == "add":
        rows = plan.get("new_rows") or []
        if not rows:
            return df, "No records found to add.", False
        try:
            add_df = pd.DataFrame(rows).reindex(columns=df.columns)
            out = pd.concat([df, add_df], ignore_index=True)
        except Exception as e:
            return df, f"Could not add records: {e}", False
        return out, f"Added {len(add_df)} new record(s). Total now {len(out)} rows.", True

    if action == "add_column":
        name = plan.get("new_column")
        formula = plan.get("formula")
        if not name or not formula:
            return df, "Need both a column name and a formula.", False

        # Guard rail: refuse Excel-style formulas we can't safely translate.
        # This prevents "silent miscompute" (e.g. INDEX/MATCH/MAX/AVERAGE
        # getting reduced by the LLM to a plain per-row col1/col2).
        bad = _unsupported_functions(user_prompt, formula)
        if bad:
            fn_list = ", ".join(bad)
            msg = (
                f"Could not apply this formula safely - it uses Excel "
                f"functions ({fn_list}) that this app's per-row formula "
                f"engine doesn't support yet.\n\n"
                f"What works today: per-row arithmetic on existing columns, "
                f"e.g.  `total = price * quantity`  or  `bmi = weight / (height ** 2)`.\n\n"
                f"For aggregate values (max, average, sum, lookups, IFs) "
                f"please ask the question directly in chat instead, e.g. "
                f"\"what is the max of column A?\" or \"average of column B where D > 10\"."
            )
            return df, msg, False

        out = df.copy()
        try:
            out[name] = _eval_formula(out, formula)
        except Exception as e:
            return df, f"Could not apply formula '{formula}': {e}", False
        return out, f"Added column '{name}' = {formula}.", True

    if action == "delete_column":
        col = _resolve_col(plan.get("target_column"), cols)
        if not col:
            return df, "Could not find that column to delete.", False
        return df.drop(columns=[col]), f"Deleted column '{col}'.", True

    if action == "rename_column":
        old = _resolve_col(plan.get("target_column"), cols)
        new = plan.get("new_column")
        if not old or not new:
            return df, "Need the existing column and a new name.", False
        return df.rename(columns={old: new}), f"Renamed '{old}' to '{new}'.", True

    if action == "search":
        q = str(plan.get("query", "")).strip()
        if not q:
            return df, "No search text found.", False
        target_cols = [c for c in (plan.get("columns") or []) if _resolve_col(c, cols)]
        target_cols = [_resolve_col(c, cols) for c in target_cols] or cols
        mask = pd.Series(False, index=df.index)
        for c in target_cols:
            mask |= df[c].astype(str).str.contains(q, case=False, na=False)
        out = df[mask]
        return out, f"Found {len(out)} row(s) for '{q}'.", True

    if action == "filter":
        out = df
        applied = []
        for cond in plan.get("conditions", []) or []:
            col = _resolve_col(cond.get("column"), cols)
            if col is None:
                continue
            op = cond.get("op", "==")
            val = cond.get("value")
            try:
                out = _apply_condition(out, col, op, val)
                applied.append(f"{col} {op} {val}")
            except Exception as e:
                return df, f"Could not apply filter ({col} {op} {val}): {e}", False
        if not applied:
            return df, "No valid filter condition found.", False
        return out, f"Filter applied: {', '.join(applied)} - {len(out)} row(s).", True

    if action == "sort":
        by = [_resolve_col(c, cols) for c in (plan.get("sort_by") or [])]
        by = [c for c in by if c]
        if not by:
            return df, "No valid column to sort by.", False
        asc = plan.get("ascending")
        if not isinstance(asc, list) or len(asc) != len(by):
            asc = [True] * len(by)
        out = df.sort_values(by=by, ascending=asc, kind="stable")
        order = "ascending" if all(asc) else ("descending" if not any(asc) else "mixed")
        return out, f"Sorted by {', '.join(map(str, by))} ({order}).", True

    # action == "answer" -> no data operation, just an LLM answer
    return df, None, False


def _apply_condition(df, col, op, val):
    """Apply a single filter condition (used by the "filter" action)."""
    series = df[col]
    # Coerce to numeric if the target value looks numeric but the column is not
    if isinstance(val, (int, float)) and not pd.api.types.is_numeric_dtype(series):
        series_num = pd.to_numeric(series, errors="coerce")
    else:
        series_num = series

    if op == "==":
        return (df[series.astype(str).str.lower() == str(val).lower()]
                if not pd.api.types.is_numeric_dtype(series)
                else df[series_num == val])
    if op == "!=":
        return (df[series.astype(str).str.lower() != str(val).lower()]
                if not pd.api.types.is_numeric_dtype(series)
                else df[series_num != val])
    if op == ">":
        return df[series_num > val]
    if op == "<":
        return df[series_num < val]
    if op == ">=":
        return df[series_num >= val]
    if op == "<=":
        return df[series_num <= val]
    if op == "contains":
        return df[series.astype(str).str.contains(str(val), case=False, na=False)]
    if op == "startswith":
        return df[series.astype(str).str.lower().str.startswith(str(val).lower(), na=False)]
    if op == "endswith":
        return df[series.astype(str).str.lower().str.endswith(str(val).lower(), na=False)]
    raise ValueError(f"Unknown operator: {op}")


def data_summary(df: pd.DataFrame, max_rows=40) -> str:
    """Build a compact summary of a DataFrame for the LLM's context."""
    lines = [f"Rows: {len(df)}, Columns: {len(df.columns)}"]
    lines.append("Columns & types:")
    for c in df.columns:
        lines.append(f"  - {c} ({df[c].dtype})")
    lines.append("\nFirst rows (CSV):")
    lines.append(df.head(max_rows).to_csv(index=False))
    return "\n".join(lines)
