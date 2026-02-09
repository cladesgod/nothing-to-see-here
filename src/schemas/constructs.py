"""AAAW (Attitudes Toward the Use of AI in the Workplace) construct definitions.

Based on Park et al. (2024) as used in Lee et al. (2025).
The AAAW scale measures six dimensions of attitudes toward AI in the workplace.

Each dimension includes two orbiting dimensions for content validity assessment
using the Colquitt et al. (2019) method: the content reviewer rates the item's
relevance to the target dimension AND to two related (orbiting) dimensions,
then c-value and d-value are computed from the difference.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ConstructDimension(BaseModel):
    """A sub-dimension of a psychological construct."""

    name: str
    definition: str
    example_items: list[str] = Field(default_factory=list)
    orbiting_dimensions: list[str] = Field(
        default_factory=list,
        description="Names of two related dimensions used for d-value calculation.",
    )


class Construct(BaseModel):
    """A psychological construct with its dimensions."""

    name: str
    definition: str
    dimensions: list[ConstructDimension]

    def get_dimension(self, name: str) -> ConstructDimension | None:
        """Get a dimension by name."""
        for d in self.dimensions:
            if d.name == name:
                return d
        return None

    def get_orbiting_definitions(self, dimension_name: str) -> list[tuple[str, str]]:
        """Return (name, definition) pairs for the orbiting dimensions.

        Used by the content reviewer to present orbiting constructs for rating.
        """
        dim = self.get_dimension(dimension_name)
        if dim is None:
            return []
        result = []
        for orb_name in dim.orbiting_dimensions:
            orb_dim = self.get_dimension(orb_name)
            if orb_dim:
                result.append((orb_dim.name, orb_dim.definition))
        return result


# ---------------------------------------------------------------------------
# AAAW 6 Dimensions (Park et al., 2024)
# ---------------------------------------------------------------------------

AAAW_DIMENSIONS = [
    ConstructDimension(
        name="AI Use Anxiety",
        definition=(
            "The degree of apprehension or worry individuals experience when "
            "interacting with or contemplating the use of AI systems in their "
            "work environment."
        ),
        example_items=[
            "I feel uneasy when I have to use AI tools at work.",
            "The thought of relying on AI for work tasks makes me nervous.",
        ],
        orbiting_dimensions=["Job Insecurity", "Personal Utility"],
    ),
    ConstructDimension(
        name="Personal Utility",
        definition=(
            "The perceived usefulness and practical value of AI tools for "
            "enhancing one's own job performance and productivity."
        ),
        example_items=[
            "AI tools help me accomplish my work tasks more efficiently.",
            "I find AI applications useful for my daily work activities.",
        ],
        orbiting_dimensions=["Perceived Quality of AI", "Perceived Adaptability of AI"],
    ),
    ConstructDimension(
        name="Perceived Humanlikeness of AI",
        definition=(
            "The extent to which individuals perceive AI systems as possessing "
            "human-like qualities such as understanding, empathy, or social presence."
        ),
        example_items=[
            "AI systems seem to understand my needs like a human colleague would.",
            "Interacting with AI feels similar to interacting with a person.",
        ],
        orbiting_dimensions=["Perceived Adaptability of AI", "Perceived Quality of AI"],
    ),
    ConstructDimension(
        name="Perceived Adaptability of AI",
        definition=(
            "The degree to which individuals believe AI systems can flexibly "
            "adjust to varying tasks, contexts, and user needs in the workplace."
        ),
        example_items=[
            "AI tools can easily adapt to different types of work tasks.",
            "AI systems adjust well to my changing work requirements.",
        ],
        orbiting_dimensions=["Personal Utility", "Perceived Humanlikeness of AI"],
    ),
    ConstructDimension(
        name="Perceived Quality of AI",
        definition=(
            "The evaluation of the overall reliability, accuracy, and output "
            "quality of AI systems as experienced in the work context."
        ),
        example_items=[
            "The outputs produced by AI tools at work are reliable.",
            "AI systems consistently deliver high-quality results.",
        ],
        orbiting_dimensions=["Personal Utility", "Perceived Adaptability of AI"],
    ),
    ConstructDimension(
        name="Job Insecurity",
        definition=(
            "The perceived threat that AI technologies pose to one's job "
            "stability, career prospects, or professional relevance."
        ),
        example_items=[
            "I worry that AI might replace my role in the organization.",
            "AI advancements make me uncertain about my future career prospects.",
        ],
        orbiting_dimensions=["AI Use Anxiety", "Personal Utility"],
    ),
]


AAAW_CONSTRUCT = Construct(
    name="Attitudes Toward the Use of AI in the Workplace (AAAW)",
    definition=(
        "A multidimensional construct measuring individuals' attitudes toward "
        "artificial intelligence in their work environment, encompassing anxiety, "
        "utility perceptions, humanlikeness attributions, adaptability beliefs, "
        "quality evaluations, and job insecurity concerns (Park et al., 2024)."
    ),
    dimensions=AAAW_DIMENSIONS,
)
