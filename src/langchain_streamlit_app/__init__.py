"""Job Description Analyser: logic package (no Streamlit imports)."""

from langchain_streamlit_app.schemas import (
    InterviewTopic,
    JobAnalysis,
    Skill,
    StudyPlanItem,
)

__all__ = ["InterviewTopic", "JobAnalysis", "Skill", "StudyPlanItem", "main"]


def main() -> None:
    print("Hello from langchain-streamlit-app!")
