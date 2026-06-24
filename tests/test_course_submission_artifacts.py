import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "certilab_agentic_rag_demo.ipynb"
README_PATH = PROJECT_ROOT / "README.md"
NOTEBOOKS_README_PATH = PROJECT_ROOT / "notebooks" / "README.md"
COURSE_ARTICLE_TITLE = "Building an Adaptive RAG System with LangGraph, OpenAI and Tavily"
COURSE_ARTICLE_URL = (
    "https://levelup.gitconnected.com/"
    "building-an-adaptive-rag-system-with-langgraph-openai-and-tavily-c4ee39d2f021"
)

REQUIRED_README_MARKERS = [
    "## Entrega académica",
    "LevelUp",
    "notebooks/certilab_agentic_rag_demo.ipynb",
    "Checklist de entrega",
    "uv run pytest",
    "uv run ruff check .",
    "uv run mypy app",
]

REQUIRED_NOTEBOOK_MARKERS = [
    "mock-only",
    "offline",
    "X-Demo-Token",
    "structured",
    "semantic",
    "combined",
    "source_id",
]

UNSAFE_NOTEBOOK_MARKERS = [
    "OPENAI_API_KEY",
    "TAVILY_API_KEY",
    "AWS_SECRET_ACCESS_KEY",
    "MYSQL_READONLY_DSN",
    "PHOENIX_COLLECTOR_ENDPOINT",
    "load_dotenv",
    "%env",
    "requests.",
    "httpx.",
    "openai.",
    "tavily",
    "/Users/",
    "C:\\Users\\",
    "data/pdf/",
]


def _notebook_text(notebook: dict[str, Any]) -> str:
    cells = notebook.get("cells", [])
    return "\n".join("".join(cell.get("source", [])) for cell in cells)


def test_offline_demo_notebook_is_valid_json_and_documents_mock_flow() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) >= 4

    text = _notebook_text(notebook)
    for marker in REQUIRED_NOTEBOOK_MARKERS:
        assert marker in text


def test_readmes_include_submission_checklist_notebook_link_and_verification_commands() -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    notebooks_readme = NOTEBOOKS_README_PATH.read_text(encoding="utf-8")

    for marker in REQUIRED_README_MARKERS:
        assert marker in readme

    assert "certilab_agentic_rag_demo.ipynb" in notebooks_readme
    assert "offline" in notebooks_readme
    assert "mock" in notebooks_readme
    assert "secretos" in notebooks_readme


def test_public_submission_artifacts_reference_professor_article_exactly() -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    notebooks_readme = NOTEBOOKS_README_PATH.read_text(encoding="utf-8")

    for artifact_text in (readme, notebooks_readme):
        assert COURSE_ARTICLE_TITLE in artifact_text
        assert COURSE_ARTICLE_URL in artifact_text

    assert "El título/URL exacto no está incluido" not in readme


def test_course_submission_artifacts_do_not_contain_unsafe_notebook_markers() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    combined_text = _notebook_text(notebook)
    combined_text += "\n" + NOTEBOOKS_README_PATH.read_text(encoding="utf-8")
    combined_text = combined_text.replace(COURSE_ARTICLE_URL, "")

    for marker in UNSAFE_NOTEBOOK_MARKERS:
        assert marker not in combined_text
