"""RAG quality evaluation using Phoenix experiments.

This module provides the shared evaluation logic used by:
- ``scripts/eval_rag.py`` (CLI entry point)
- ``app/main.py`` (POST /admin/evaluate endpoint)
"""

from __future__ import annotations

import logging
import os
from typing import Any

from app.config import Settings
from app.domain.models import Role
from app.security.access_control import Principal

logger = logging.getLogger(__name__)

_TEST_CASES: list[dict[str, str]] = [
    {"question": "¿cuántos certificados hay?", "expected_contains": "177"},
    {"question": "¿qué certificados están pendientes?", "expected_contains": "pendiente"},
    {"question": "certificados del cliente 3", "expected_contains": "cliente"},
    {"question": "¿qué es CERT-QA-1777623761?", "expected_contains": "certificado"},
]


def _admin_principal() -> Principal:
    return Principal(role=Role.ADMIN, customer_id=None, user_id=1)


def _check_phoenix(base_url: str) -> Any:
    """Check that Phoenix is reachable, or raise a clear error."""
    from phoenix.client import Client as PhoenixClient

    try:
        client = PhoenixClient(base_url=base_url)
        _ = client.datasets.list()
        return client
    except Exception as exc:
        raise RuntimeError(
            f"Cannot connect to Phoenix at {base_url!r}: {exc}. "
            "Make sure Phoenix is running (e.g. docker compose up phoenix)."
        ) from exc


def _make_dataset(client: Any, settings: Settings) -> Any:
    """Return existing dataset or create a new one with test cases."""
    ds_name = settings.phoenix_project_name + "-eval"

    # The list endpoint returns plain dicts; look for a matching name
    datasets: list[dict[str, Any]] = list(client.datasets.list())
    for ds_dict in datasets:
        if ds_dict.get("name") == ds_name:
            existing_id: str = ds_dict["id"]
            # get_dataset returns a proper Dataset object with version_id etc.
            dataset = client.datasets.get_dataset(dataset=existing_id)
            logger.info("Using existing dataset '%s' (%s)", ds_name, existing_id)
            return dataset

    logger.info("Creating dataset '%s' with %d examples", ds_name, len(_TEST_CASES))
    return client.datasets.create_dataset(
        name=ds_name,
        inputs=[{"question": tc["question"]} for tc in _TEST_CASES],
        outputs=[{"expected_contains": tc["expected_contains"]} for tc in _TEST_CASES],
        dataset_description="Golden test set for Certilab RAG quality evaluation",
    )


def _make_task(settings: Settings) -> Any:
    """Build a task callable that runs the RAG pipeline for one question.

    Forces mock mode so the evaluation never requires external databases or
    services — the mock pipeline reads from JSON fixtures.
    """
    eval_settings = settings.model_copy(update={"app_mode": "mock"})

    def task(example: dict[str, Any]) -> dict[str, Any]:
        from app.graph import build_rag_pipeline

        question = example["input"]["question"]
        pipeline = build_rag_pipeline(eval_settings)
        response = pipeline.ask(question, _admin_principal())
        return {
            "answer": response.answer,
            "route": response.route,
            "sources": [s.model_dump() for s in response.sources],
        }

    return task


def _make_evaluators() -> list[Any]:
    """Return a list of evaluator callables for the experiment."""
    from phoenix.client.resources.experiments.evaluators import create_evaluator

    # ── CODE evaluator: expected text match ───────────────────────────────

    def _safe_output(output: Any) -> dict[str, Any]:
        """Return output if it is a dict, else an empty dict."""
        return output if isinstance(output, dict) else {}

    @create_evaluator(kind="CODE", name="Expected Contains")
    def expected_contains(
        output: dict[str, Any] | None,
        expected: dict[str, Any],
    ) -> tuple[float, str]:
        """Check if the answer text contains the expected substring."""
        out = _safe_output(output)
        answer = out.get("answer", "")
        expected_text = expected.get("expected_contains", "")
        if expected_text in answer:
            return 1.0, f"Answer contains {expected_text!r}"
        return 0.0, f"Answer does not contain {expected_text!r}"

    # ── LLM evaluator: Q&A Relevance ──────────────────────────────────────

    @create_evaluator(kind="LLM", name="Q&A Relevance")
    def qna_relevance(
        output: dict[str, Any] | None,
        input: dict[str, Any],
    ) -> tuple[float, str]:
        """Ask an LLM whether the answer actually addresses the question."""
        out = _safe_output(output)
        if not out.get("answer"):
            return 0.0, "No answer provided — cannot evaluate relevance"
        try:
            from openai import OpenAI

            question = input.get("question", "")
            answer = out["answer"][:2000]
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Question: {question}\n\nAnswer: {answer}\n\n"
                            "Rate how well the answer addresses the question from 0.0 to 1.0.\n"
                            "Output only a single number between 0 and 1."
                        ),
                    }
                ],
                temperature=0.0,
            )
            score_str = (resp.choices[0].message.content or "0").strip()
            score = max(0.0, min(1.0, float(score_str)))
            return score, f"Q&A Relevance score: {score:.2f}"
        except Exception as exc:
            return 0.0, f"Q&A Relevance error: {exc}"

    # ── LLM evaluator: Hallucination ──────────────────────────────────────

    @create_evaluator(kind="LLM", name="Hallucination")
    def hallucination(output: dict[str, Any] | None) -> tuple[float, str]:
        """Ask an LLM whether the answer is grounded in the retrieved sources."""
        out = _safe_output(output)
        if not out.get("answer"):
            return 0.0, "No answer provided — cannot evaluate hallucination"
        try:
            from openai import OpenAI

            answer = out["answer"][:2000]
            sources_list = out.get("sources") or []
            sources_text = "\n".join(
                s.get("snippet", "") for s in sources_list
            )[:3000]

            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Context:\n{sources_text}\n\nResponse:\n{answer}\n\n"
                            "Is the response hallucinated? Reply only 'factual' or 'hallucinated'."
                        ),
                    }
                ],
                temperature=0.0,
            )
            label = (resp.choices[0].message.content or "").strip().lower()
            if "hallucinated" in label:
                return 0.0, "Response is hallucinated"
            return 1.0, "Response is factual (grounded in sources)"
        except Exception as exc:
            return 0.0, f"Hallucination evaluator error: {exc}"

    # ── LLM evaluator: Context Relevance ──────────────────────────────────

    @create_evaluator(kind="LLM", name="Context Relevance")
    def context_relevance(
        output: dict[str, Any] | None,
        input: dict[str, Any],
    ) -> tuple[float, str]:
        """Ask an LLM whether the retrieved documents help answer the question."""
        out = _safe_output(output)
        if not out.get("sources"):
            return 0.0, "No sources retrieved — cannot evaluate context relevance"
        try:
            from openai import OpenAI

            question = input.get("question", "")
            sources_list = out.get("sources") or []
            sources_text = "\n".join(
                s.get("snippet", "") for s in sources_list
            )[:3000]

            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Question: {question}\n\n"
                            f"Retrieved documents:\n{sources_text}\n\n"
                            "Rate how relevant the retrieved documents are to answering "
                            "the question from 0.0 to 1.0. Output only a single number."
                        ),
                    }
                ],
                temperature=0.0,
            )
            score_str = (resp.choices[0].message.content or "0").strip()
            score = max(0.0, min(1.0, float(score_str)))
            return score, f"Context Relevance score: {score:.2f}"
        except Exception as exc:
            return 0.0, f"Context Relevance error: {exc}"

    return [expected_contains, qna_relevance, hallucination, context_relevance]


def run_evaluation(settings: Settings) -> dict[str, Any]:
    """Run the full RAG evaluation experiment against Phoenix.

    Creates (or reuses) a Phoenix dataset with golden test questions, runs
    each question through the RAG pipeline, and evaluates the results with
    both code and LLM-based evaluators.

    Returns a summary dict with:
        ``evaluated`` — number of test cases processed
        ``experiment_id`` — Phoenix experiment ID for UI viewing
        ``scores`` — dict of evaluator name -> list of scores
    """
    phoenix_url = settings.phoenix_collector_endpoint.replace("/v1/traces", "")
    client = _check_phoenix(phoenix_url)

    dataset = _make_dataset(client, settings)
    task_fn = _make_task(settings)
    evaluators = _make_evaluators()

    experiment = client.experiments.run_experiment(
        dataset=dataset,
        task=task_fn,
        evaluators=evaluators,
        experiment_name="RAG Quality Evaluation",
        experiment_description="Automated RAG quality measurement using golden test cases",
        print_summary=False,
    )

    # Aggregate scores from evaluation runs
    scores: dict[str, list[float]] = {}
    eval_runs: list[Any] = list(experiment.get("evaluation_runs", []))
    for eval_run in eval_runs:
        name: str = getattr(eval_run, "name", "?")
        result: Any = getattr(eval_run, "result", None)
        if result is None:
            continue
        if isinstance(result, dict):
            score_val = result.get("score")
        elif isinstance(result, list):
            score_val = result[0].get("score") if result else None
        else:
            score_val = None
        if score_val is not None:
            scores.setdefault(name, []).append(float(score_val))

    summary: dict[str, Any] = {
        "evaluated": len(experiment.get("task_runs", [])),
        "experiment_id": str(experiment.get("experiment_id", "")),
        "scores": {k: {"values": v, "avg": round(sum(v) / len(v), 3)} for k, v in scores.items()},
    }
    return summary
