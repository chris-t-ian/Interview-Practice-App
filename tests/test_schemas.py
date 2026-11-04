import pytest
from pydantic import ValidationError

from langchain_streamlit_app import JobAnalysis

SAMPLE = {
    "role_title": "Junior Data Analyst",
    "seniority": "junior",
    "skills": [
        {
            "name": "SQL",
            "category": "technical",
            "priority": "must-have",
            "evidence": "Strong SQL skills required",
        },
        {
            "name": "Tableau",
            "category": "tool",
            "priority": "nice-to-have",
            "evidence": "Experience with Tableau is a plus",
        },
    ],
    "interview_topics": [
        {
            "topic": "SQL querying",
            "why_likely": "SQL is listed as the core requirement.",
            "example_questions": [
                "Explain the difference between INNER and LEFT JOIN.",
                "How would you find duplicate rows in a table?",
            ],
            "difficulty": "medium",
        }
    ],
    "study_plan": [
        {
            "day_or_block": "Day 1",
            "focus": "SQL joins and aggregations",
            "activities": ["Solve 5 join exercises", "Review GROUP BY and HAVING"],
            "resources": ["official PostgreSQL docs", "an online SQL practice set"],
            "estimated_hours": 2.5,
        }
    ],
}


def test_sample_job_analysis_validates():
    analysis = JobAnalysis.model_validate(SAMPLE)

    assert analysis.role_title == "Junior Data Analyst"
    assert analysis.skills[0].priority == "must-have"
    assert len(analysis.interview_topics[0].example_questions) == 2
    assert analysis.study_plan[0].estimated_hours == 2.5
    assert analysis.reasoning is None


def test_too_few_example_questions_is_rejected():
    bad = {**SAMPLE, "interview_topics": [{**SAMPLE["interview_topics"][0], "example_questions": ["Only one?"]}]}
    with pytest.raises(ValidationError):
        JobAnalysis.model_validate(bad)
