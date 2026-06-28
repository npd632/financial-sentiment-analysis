"""Company-relevance filter for news headlines."""

from __future__ import annotations

import re

import yaml


def load_ticker_aliases(path: str) -> dict[str, list[str]]:
    """Load ticker -> alias list mapping from YAML."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return {str(k).upper(): [str(a).lower() for a in v] for k, v in raw.items()}


def _token_pattern(token: str) -> re.Pattern[str]:
    """Word-boundary regex for a lowercase token (supports multi-word aliases)."""
    escaped = re.escape(token.lower())
    return re.compile(rf"(?<!\w){escaped}(?!\w)")


def is_company_relevant(
    cleaned_text: str,
    ticker: str,
    aliases: dict[str, list[str]],
) -> bool:
    """
    Return True if headline text mentions the ticker or a configured alias.

    Uses word-boundary matching on cleaned lowercase text.
    """
    if not cleaned_text:
        return False

    text = cleaned_text.lower()
    ticker_lower = ticker.lower()

    if _token_pattern(ticker_lower).search(text):
        return True

    for alias in aliases.get(ticker.upper(), []):
        if _token_pattern(alias).search(text):
            return True

    return False


def apply_company_filter(
    df,
    aliases_path: str,
    text_col: str = "cleaned_text",
    ticker_col: str = "stock",
):
    """Add headline_relevant column and return filtered stats."""
    aliases = load_ticker_aliases(aliases_path)
    relevant = df.apply(
        lambda row: is_company_relevant(
            row[text_col], row[ticker_col], aliases
        ),
        axis=1,
    )
    df = df.copy()
    df["headline_relevant"] = relevant
    return df, aliases
