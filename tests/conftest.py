"""Shared test fixtures."""

from __future__ import annotations

import os

# Ensure tests don't accidentally call real APIs
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")
os.environ.setdefault("HF_TOKEN", "test-key")
os.environ.setdefault("GROQ_API_KEY", "test-key")
