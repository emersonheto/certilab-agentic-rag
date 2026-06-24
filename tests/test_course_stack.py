import builtins
import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from app.config import Settings
from app.domain.models import RetrievedSource, Role
from app.graph import CertilabRagPipeline, build_rag_pipeline
from app.retrieval.grader import route_question
from app.security.access_control import Principal
from app.tools.openai_client import AnswerGenerator
from app.tools.web_search import TavilyWebSearch, WebSearchConfig


ADMIN = Principal(role=Role.ADMIN, customer_id=None, user_id=1)


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = FakeMessage(content)


class FakeChatCompletions:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    def create(self, **kwargs: Any) -> object:
        self.messages = kwargs["messages"]
        return type("FakeResponse", (), {"choices": [FakeChoice("Respuesta generada por mock OpenAI.")]})()


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.completions = FakeChatCompletions()
        self.chat = type("FakeChat", (), {"completions": self.completions})()


class FakeOpenAIAdapter:
    chat_model = "fake-chat-model"

    def __init__(self) -> None:
        self.fake_client = FakeOpenAIClient()

    def client(self) -> FakeOpenAIClient:
        return self.fake_client


class FakeTavilyClient:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, *, query: str, max_results: int) -> dict[str, list[dict[str, str]]]:
        self.queries.append(query)
        return {
            "results": [
                {
                    "title": "Public calibration update",
                    "url": "https://example.test/calibration",
                    "content": f"Public summary for {len(query)} characters, limited to {max_results} results.",
                }
            ]
        }


def test_route_selection_includes_course_stack_paths() -> None:
    assert route_question("¿Cuántos certificados hay?") == "structured"
    assert route_question("Resumen del procedimiento técnico") == "semantic"
    assert route_question("Estado del certificado y resumen del procedimiento") == "combined"
    assert route_question("Buscar contexto web externo") == "web_search"


def test_langgraph_engine_falls_back_when_optional_dependency_is_unavailable(monkeypatch: Any) -> None:
    sys.modules.pop("app.graph_langgraph", None)
    original_import = builtins.__import__

    def fail_langgraph_pipeline_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "app.graph_langgraph":
            raise ImportError(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_langgraph_pipeline_import)

    pipeline = build_rag_pipeline(Settings(_env_file=None, data_dir=Path("data"), graph_engine="langgraph"))

    response = pipeline.ask("¿Cuántos certificados hay?", ADMIN)
    assert isinstance(pipeline, CertilabRagPipeline)
    assert response.route == "structured"


def test_langgraph_route_parity_for_e2e_paths() -> None:
    pytest.importorskip("langgraph")
    from app.graph_langgraph import CertilabLangGraphPipeline

    settings = Settings(_env_file=None, data_dir=Path("data"), graph_engine="langgraph", tavily_api_key=None)
    deterministic = CertilabRagPipeline(Path("data"), settings=settings)
    langgraph_pipeline = CertilabLangGraphPipeline(Path("data"), settings=settings)
    questions = [
        "¿Cuántos certificados hay?",
        "Resumen del procedimiento técnico",
        "Estado del certificado y resumen del procedimiento",
        "Buscar contexto web externo público",
    ]

    for question in questions:
        deterministic_response = deterministic.ask(question, ADMIN)
        langgraph_response = langgraph_pipeline.ask(question, ADMIN)
        assert langgraph_response.route == deterministic_response.route
        assert len(langgraph_response.sources) == len(deterministic_response.sources)


def test_deterministic_pipeline_handles_web_search_fallback_without_tavily_key() -> None:
    pipeline = CertilabRagPipeline(Path("data"), settings=Settings(_env_file=None, tavily_api_key=None))

    response = pipeline.ask("Necesito contexto web externo", ADMIN)

    assert response.route == "web_search"
    assert response.sources[0].source_type == "web_search"
    assert "Tavily API key is not configured" in response.answer


def test_tavily_adapter_uses_mocked_client_successfully() -> None:
    search = TavilyWebSearch(WebSearchConfig(tavily_api_key="fake-key"))
    fake_client = FakeTavilyClient()
    search._client = fake_client

    results = search.search("public calibration news token=unsafe")

    assert results[0].source_note == "fallback"
    assert fake_client.queries == []


def test_tavily_adapter_sanitizes_safe_generic_query() -> None:
    search = TavilyWebSearch(WebSearchConfig(tavily_api_key="fake-key"))
    fake_client = FakeTavilyClient()
    search._client = fake_client

    results = search.search("public calibration standards news")

    assert results[0].title == "Public calibration update"
    assert results[0].source_note == "tavily"
    assert fake_client.queries == ["public calibration standards news"]


def test_tavily_adapter_blocks_certificate_codes_urls_paths_and_customer_terms() -> None:
    search = TavilyWebSearch(WebSearchConfig(tavily_api_key="fake-key"))
    fake_client = FakeTavilyClient()
    search._client = fake_client

    unsafe_queries = [
        "buscar CERT-2025-001 en web",
        "buscar https://internal.example.test/certificados",
        "buscar data/mock/certificates.json",
        "buscar cliente 101 certificado",
    ]

    for query in unsafe_queries:
        result = search.search(query)
        assert result[0].source_note == "fallback"
    assert fake_client.queries == []


def test_openai_generator_falls_back_without_key() -> None:
    generator = AnswerGenerator(Settings(_env_file=None, app_mode="real", openai_api_key=None))

    answer = generator.generate("full question stays local", [], "deterministic answer")

    assert answer == "deterministic answer"


def test_openai_generator_uses_mocked_adapter_in_real_mode() -> None:
    adapter = FakeOpenAIAdapter()
    source = RetrievedSource(
        certificate_id=1,
        code="CERT-2025-001",
        customer_id=101,
        source_type="metadata",
        path="data/mock/certificates.json",
        snippet=(
            "Raw CERT-2025-001 token=super-secret mysql://user:pass@localhost/db "
            "data/mock/certificates.json cliente 101"
        ),
    )
    generator = AnswerGenerator(
        Settings(_env_file=None, app_mode="real", openai_api_key="fake-key"), adapter=adapter
    )

    answer = generator.generate("full question stays local", [source], "deterministic answer")

    assert answer == "Respuesta generada por mock OpenAI."
    prompt = adapter.fake_client.completions.messages[1]["content"]
    assert "CERT-2025-001" not in prompt
    assert "super-secret" not in prompt
    assert "mysql://" not in prompt
    assert "data/mock" not in prompt
    assert "cliente 101" not in prompt
    assert "[REDACTED_CERTIFICATE_CODE]" in prompt


def test_chainlit_module_import_does_not_require_chainlit_runtime(monkeypatch: Any) -> None:
    class FakeChainlitModule:
        Message = FakeMessage

        @staticmethod
        def on_message(func: object) -> object:
            return func

    monkeypatch.setitem(sys.modules, "chainlit", FakeChainlitModule())
    sys.modules.pop("ui.chainlit_app", None)

    module = importlib.import_module("ui.chainlit_app")

    assert hasattr(module, "main")


def test_chainlit_demo_prefers_explicit_or_client_token() -> None:
    module = importlib.import_module("ui.chainlit_app")
    settings = Settings(
        _env_file=None,
        chainlit_demo_token="explicit-token",
        demo_admin_token="admin-token",
        demo_client_101_token="client-token",
    )

    assert module._default_demo_token(settings) == "explicit-token"

    client_settings = Settings(_env_file=None, demo_admin_token="admin-token", demo_client_101_token="client-token")
    assert module._default_demo_token(client_settings) == "client-token"

    admin_only_settings = Settings(
        _env_file=None,
        demo_admin_token="admin-token",
        demo_client_101_token=None,
        demo_client_202_token=None,
    )
    assert module._default_demo_token(admin_only_settings) is None
