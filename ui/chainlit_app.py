"""Chainlit application handlers for the Certilab RAG chat UI.

This module registers all Chainlit lifecycle handlers: chat profiles,
chat start, starter questions, on_message (the main Q&A handler with
progressive ``cl.Step`` per pipeline phase), and admin action callbacks
(reindex, evaluate, refresh status).

Usage
-----
    chainlit run ui/chainlit_app.py

The module is safe to import when Chainlit is not installed — it degrades
to an ``ImportError``-raising fallback.
"""

from __future__ import annotations

import asyncio
from time import perf_counter

from app.config import get_settings
from app.graph import build_rag_pipeline
from app.schemas import AskResponse
from app.security.access_control import Principal

from ui.client import PipelineClient
from ui.settings import (
    _build_principal,
    _embedding_label,
    _get_index_stats,
    _PROFILES,
    ROUTE_LABELS,
    format_answer,
)

try:
    import chainlit as cl
except ImportError:  # pragma: no cover
    cl = None  # type: ignore[assignment]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _route_label(route_name: str) -> str:
    """Return the human-readable label for a pipeline route name.

    Args:
        route_name: The internal route name (e.g. ``"structured"``).

    Returns:
        A markdown label with emoji (e.g. ``"📊 Ruta: estructurada"``).
    """
    return ROUTE_LABELS.get(route_name, f"📌 Ruta: {route_name}")


def _get_client() -> PipelineClient:
    """Build a ``PipelineClient`` for the current session's principal.

    Returns:
        A ready-to-use ``PipelineClient`` instance.

    Raises:
        RuntimeError: If no ``principal`` is stored in the Chainlit session.
    """
    principal: Principal | None = cl.user_session.get("principal")
    if principal is None:
        raise RuntimeError("Sin sesión activa. Recargá la página y elegí un perfil.")
    settings = get_settings()
    pipeline = build_rag_pipeline(settings)
    return PipelineClient(pipeline, principal)


# ── Chainlit handlers ─────────────────────────────────────────────────────────


if cl is not None:

    @cl.set_chat_profiles
    async def chat_profiles() -> list[cl.ChatProfile]:
        return [
            cl.ChatProfile(
                name=p["name"],
                markdown_description=p["markdown_description"],
                icon=p["icon"],
            )
            for p in _PROFILES
        ]

    @cl.set_starters
    async def starters() -> list[cl.Starter]:
        return [
            cl.Starter(
                label="¿Qué certificados hay?",
                message="¿Qué certificados hay?",
            ),
            cl.Starter(
                label="¿Cuántos certificados activos tiene el sistema?",
                message="¿Cuántos certificados activos tiene el sistema?",
            ),
        ]

    @cl.on_chat_start
    async def on_chat_start() -> None:
        settings = get_settings()
        profile_name = cl.user_session.get("chat_profile")

        # Map display name back to internal id
        name_to_id = {p["name"]: p["id"] for p in _PROFILES}
        profile_id = name_to_id.get(profile_name or "", "admin")

        try:
            principal = _build_principal(profile_id, settings)
            cl.user_session.set("principal", principal)
            role_label = principal.role.value.capitalize()
            cid = f" · cliente {principal.customer_id}" if principal.customer_id else ""

            if principal.role.value == "admin":
                stats = _get_index_stats(settings)
                provider = _embedding_label(settings)
                indexed_vectors = stats.get("indexed_vectors", stats.get("points", "?"))
                content = (
                    f"🛡️ **Panel de Administración — Certilab RAG**\n\n"
                    f"📊 **Estado del índice vectorial**\n"
                    f"- Colección: `{stats['collection']}`\n"
                    f"- Puntos indexados: **{stats['points']}**\n"
                    f"- Vectores indexados: **{indexed_vectors}**\n"
                    f"- Proveedor de embeddings: `{provider}`\n\n"
                    f"📋 **Cobertura de certificados**\n"
                    f"Usá el botón **Reindexar** para obtener el desglose completo:\n"
                    f"- Certificados en MySQL\n"
                    f"- Certificados con PDF en S3\n"
                    f"- PDFs extraídos correctamente\n"
                    f"- PDFs huérfanos (con ruta pero sin extraer)\n\n"
                    f"¿Qué querés hacer?"
                )
                await cl.Message(
                    content=content,
                    actions=[
                        cl.Action(name="reindex", payload={}, label="🔄 Reindexar", tooltip="Reindexar certificados desde MySQL"),
                        cl.Action(name="evaluate", payload={}, label="📊 Evaluar calidad", tooltip="Ejecutar evaluación de calidad RAG"),
                        cl.Action(name="refresh_status", payload={}, label="📊 Ver estado", tooltip="Refrescar estado del índice"),
                    ],
                ).send()
            else:
                await cl.Message(
                    content=f"Sesión iniciada como **{role_label}**{cid}. ¿En qué puedo ayudarte?",
                ).send()
        except Exception as exc:
            await cl.Message(content=f"⚠️ Error al iniciar sesión: {exc}").send()

    @cl.on_message
    async def main(message: cl.Message) -> None:
        principal: Principal | None = cl.user_session.get("principal")
        if principal is None:
            await cl.Message(content="⚠️ Sin sesión activa. Recargá la página y elegí un perfil.").send()
            return

        client = _get_client()

        # Phase 1: Route
        async with cl.Step(name="🧭 Enrutando consulta") as step:
            step.output = "Analizando la pregunta..."
            route = await client.route(message.content)
            step.output = f"Ruta seleccionada: {_route_label(route)}"

        # Short-circuit for off-topic: don't retrieve or generate
        if route == "off_topic":
            from app.pipeline.guards import OFF_TOPIC_ANSWER
            await cl.Message(content=OFF_TOPIC_ANSWER).send()
            return

        # Phase 2: Retrieve
        async with cl.Step(name="🔍 Recuperando fuentes") as step:
            step.output = "Buscando información relevante..."
            fallback_answer, sources = await client.retrieve(message.content, route)
            step.output = f"{len(sources)} fuentes encontradas"

        # Phase 3: Generate
        async with cl.Step(name="✍️ Generando respuesta") as step:
            step.output = "Procesando..."
            answer = await client.generate(message.content, sources, fallback_answer)
            step.output = "Respuesta lista"

        # Final message with rich formatting
        response = AskResponse(answer=answer, route=route, sources=sources)
        formatted = format_answer(response)
        await cl.Message(content=formatted).send()

    @cl.action_callback("reindex")
    async def on_reindex(action: cl.Action) -> None:
        settings = get_settings()

        # ── Build and send the task list ──────────────────────────────────
        task_list = cl.TaskList()
        task_list.tasks = [
            cl.Task(title="Conectando a MySQL", status=cl.TaskStatus.RUNNING),
            cl.Task(title="Cargando certificados", status=cl.TaskStatus.READY),
            cl.Task(title="Extrayendo PDFs desde S3", status=cl.TaskStatus.READY),
            cl.Task(title="Generando chunks de texto", status=cl.TaskStatus.READY),
            cl.Task(title="Indexando en Qdrant (embeddings)", status=cl.TaskStatus.READY),
        ]
        await task_list.send()

        # ── Animate steps 1-4 quickly ────────────────────────────────────
        task_list.tasks[0].status = cl.TaskStatus.DONE
        task_list.tasks[1].status = cl.TaskStatus.RUNNING
        await task_list.send()

        task_list.tasks[1].status = cl.TaskStatus.DONE
        task_list.tasks[2].status = cl.TaskStatus.RUNNING
        await task_list.send()

        task_list.tasks[2].status = cl.TaskStatus.DONE
        task_list.tasks[3].status = cl.TaskStatus.RUNNING
        await task_list.send()

        task_list.tasks[3].status = cl.TaskStatus.DONE
        task_list.tasks[4].status = cl.TaskStatus.RUNNING
        await task_list.send()

        # ── Actual reindex work in a thread ──────────────────────────────
        pipeline = build_rag_pipeline(settings)
        loop = asyncio.get_running_loop()
        start = perf_counter()
        result = await loop.run_in_executor(None, pipeline.reindex)
        duration_s = perf_counter() - start

        task_list.tasks[4].status = cl.TaskStatus.DONE
        await task_list.send()
        await task_list.remove()

        # Build detailed message from enhanced result
        indexed = result.get("indexed", 0)
        skipped = result.get("skipped", 0)
        total_certs = result.get("total_certificates", "?")
        certs_with_pdf = result.get("certificates_with_pdf", "?")
        certs_embedded = result.get("certificates_embedded", "?")
        orphaned = result.get("orphaned_pdfs", "?")

        lines = [
            "\n✅ **Reindexado completo**\n",
            "\n📊 **Chunks procesados:**\n",
            f"- Nuevos chunks indexados: **{indexed}**\n",
            f"- Chunks sin cambios (saltados): **{skipped}**\n",
        ]
        if total_certs != "?":
            lines += [
                "\n📋 **Cobertura de certificados:**\n",
                f"- Total en MySQL: **{total_certs}**\n",
                f"- Con PDF en S3: **{certs_with_pdf}**\n",
                f"- PDFs extraídos: **{certs_embedded}**\n",
                f"- PDFs huérfanos: **{orphaned}**\n",
            ]
        lines.append(f"- Tiempo: **{duration_s:.1f}s**")

        await cl.Message(content="".join(lines)).send()

    @cl.action_callback("evaluate")
    async def on_evaluate(action: cl.Action) -> None:
        settings = get_settings()

        # ── Animate progress ──────────────────────────────────────────────
        msg = await cl.Message(content="📊 **Evaluando calidad RAG**\n\n⏳ Consultando pipeline …").send()

        try:
            loop = asyncio.get_running_loop()
            from concurrent.futures import ThreadPoolExecutor
            from app.eval import run_evaluation

            start = perf_counter()
            with ThreadPoolExecutor(max_workers=1) as pool:
                summary = await loop.run_in_executor(pool, run_evaluation, settings)
            duration_s = perf_counter() - start

            evaluated = summary["evaluated"]
            scores = summary.get("scores", {})
            lines = [
                f"✅ **Evaluación completa** — {evaluated} caso(s) procesado(s) en {duration_s:.1f}s\n",
            ]
            if scores:
                lines.append("\n📊 **Resultados:**\n")
                for name, data in scores.items():
                    avg = data.get("avg", "?")
                    vals = data.get("values", [])
                    lines.append(f"- **{name}**: promedio **{avg}**  ({vals})\n")
            lines.append(f"\n🔗 Ver resultados en Phoenix (experimento `{summary.get('experiment_id', '?')}`)")

            await cl.Message(content="".join(lines)).send()

        except RuntimeError as exc:
            await cl.Message(content=f"❌ Error de evaluación: {exc}").send()

        finally:
            await msg.remove()

    @cl.action_callback("refresh_status")
    async def on_refresh_status(action: cl.Action) -> None:
        settings = get_settings()
        stats = _get_index_stats(settings)
        provider = _embedding_label(settings)
        indexed_vectors = stats.get("indexed_vectors", stats.get("points", "?"))

        content = (
            f"📊 **Estado del índice vectorial**\n"
            f"- Colección: `{stats['collection']}`\n"
            f"- Puntos indexados: **{stats['points']}**\n"
            f"- Vectores indexados: **{indexed_vectors}**\n"
            f"- Proveedor de embeddings: `{provider}`\n\n"
            f"💡 Usá **Reindexar** para actualizar desde MySQL con extracción de PDFs."
        )
        await cl.Message(content=content).send()

else:

    async def main(message: object) -> None:
        raise RuntimeError("Install the optional Chainlit dependency to run the UI.")
