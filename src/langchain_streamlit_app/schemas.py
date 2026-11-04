"""Pydantic models for the structured job-description analysis.

The Field descriptions are part of the JSON schema sent to the model as the
structured-output format, so they double as instructions. The models are
kept flat (lists of simple objects, no deep nesting or unions) so that
structured output stays reliable across providers.
"""

from typing import Literal

from pydantic import BaseModel, Field

SkillCategory = Literal["technical", "soft", "domain", "tool"]
SkillPriority = Literal["must-have", "nice-to-have"]
Difficulty = Literal["easy", "medium", "hard"]
Seniority = Literal["intern", "junior", "mid", "senior", "lead", "unknown"]


class Skill(BaseModel):
    """A skill the job description asks for."""

    name: str = Field(description="Short skill name, e.g. 'Python' or 'Stakeholder communication'.")
    category: SkillCategory = Field(
        description=(
            "'technical' for programming/engineering skills, 'soft' for interpersonal skills, "
            "'domain' for industry or subject knowledge, 'tool' for specific software/platforms."
        )
    )
    priority: SkillPriority = Field(
        description="'must-have' if the JD requires it, 'nice-to-have' if it is a plus or preferred."
    )
    evidence: str = Field(
        description="Short quote or close paraphrase from the JD that shows this skill is wanted."
    )
    inferred: bool = Field(
        default=False,
        description=(
            "False if the JD names this skill explicitly; True if it is only clearly implied "
            "(e.g. 'REST APIs' implied by 'build backend services')."
        ),
    )


class InterviewTopic(BaseModel):
    """A topic that is likely to come up in the interview."""

    topic: str = Field(description="Name of the interview topic.")
    why_likely: str = Field(
        description="One or two sentences on why this topic is likely, based on the JD."
    )
    example_questions: list[str] = Field(
        min_length=2,
        max_length=4,
        description="2 to 4 realistic interview questions on this topic.",
    )
    difficulty: Difficulty = Field(
        description="Expected difficulty of these questions for the role's seniority."
    )


class StudyPlanItem(BaseModel):
    """One block of the study plan."""

    day_or_block: str = Field(description="Label for this block, e.g. 'Day 1' or 'Block 2'.")
    focus: str = Field(description="The main topic or skill this block covers.")
    activities: list[str] = Field(
        description="Concrete things to do in this block, e.g. 'Solve 3 SQL join exercises'."
    )
    resources: list[str] = Field(
        description=(
            "Generic resource types or well-known resource names, e.g. 'official Python docs' "
            "or 'a LeetCode-style practice set'. Do not invent URLs."
        )
    )
    estimated_hours: float = Field(
        gt=0, le=24, description="Estimated hours needed for this block."
    )


class JobAnalysis(BaseModel):
    """Full analysis of a job description for interview preparation."""

    # First field on purpose: structured output is generated in field order, so
    # reasoning (when requested) is written before the answer fields.
    reasoning: str | None = Field(
        default=None,
        description=(
            "Optional reasoning or review notes behind the analysis. Fill only when the "
            "instructions ask for it; otherwise leave null."
        ),
    )
    role_title: str = Field(description="Job title as stated or implied by the JD.")
    seniority: Seniority = Field(
        description="Seniority level of the role; use 'unknown' if the JD does not make it clear."
    )
    skills: list[Skill] = Field(description="Key skills from the JD, most important first.")
    interview_topics: list[InterviewTopic] = Field(
        description="Topics likely to come up in the interview, most likely first."
    )
    study_plan: list[StudyPlanItem] = Field(
        description="A short, ordered study plan to prepare for the interview topics."
    )
