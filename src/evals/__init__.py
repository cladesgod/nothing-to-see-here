"""Evaluation pipeline: independent LLM-as-a-judge quality assessment.

This module provides evaluation capabilities separate from the generation pipeline.
While the in-pipeline reviewers (content, linguistic, bias) improve items during
generation, the eval pipeline independently measures the quality of final outputs.

Key components:
- evaluators: LLM-as-judge functions (content validity, linguistic, bias, overall)
- dataset: Golden dataset management (paper Table 3) + LangSmith integration
- runner: Offline eval orchestration
"""
