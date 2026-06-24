# Ver trazas RAG con Phoenix

La observabilidad con Phoenix es opcional. Al habilitarla, la API exporta trazas OpenTelemetry para `/ask`, de modo que se pueda inspeccionar cómo el pipeline RAG decide la ruta y recupera fuentes sin registrar preguntas completas de usuarios.

## Ruta rápida

1. Instala las dependencias de desarrollo y observabilidad con `uv`:

   ```bash
   uv sync --extra dev --extra future-integrations --extra observability
   ```

2. Inicia Phoenix localmente. Si Phoenix está instalado en el entorno gestionado por `uv`, la forma preferida es:

   ```bash
   uv run phoenix serve
   ```

   Phoenix recibe trazas OTLP HTTP en `http://localhost:6006/v1/traces` y expone la interfaz en `http://localhost:6006`.

3. Habilita el tracing en `~/.config/certilab-agentic-rag/.env` o en la ruta indicada por `CERTILAB_RAG_ENV_FILE`:

   ```env
   PHOENIX_ENABLED=true
   PHOENIX_PROJECT_NAME=certilab-agentic-rag
   PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006/v1/traces
   ```

4. Ejecuta la API y envía una solicitud:

   ```bash
   uv run uvicorn app.main:app --reload
   ```

   ```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -H 'X-Demo-Token: <DEMO_ADMIN_TOKEN>' \
  -d '{"question":"<pregunta>"}'
   ```

5. Abre `http://localhost:6006` y selecciona el proyecto `certilab-agentic-rag`.

## Qué muestra Phoenix

El árbol de trazas incluye spans para el pipeline determinístico y, si `GRAPH_ENGINE=langgraph`, para el `StateGraph`:

| Span | Qué representa |
|------|----------------|
| `rag.ask` | Ejecución completa del pipeline `/ask`. |
| `rag.route_decision` | Selección determinística de ruta. |
| `rag.retrieve.structured` | Recuperación de metadatos, clientes y certificados. |
| `rag.retrieve.semantic` | Recuperación local de chunks de texto de PDFs. |
| `rag.retrieve.combined` | Flujo combinado estructurado y semántico. |
| `rag.retrieve.web_search` | Ruta web opcional con Tavily o fallback seguro. |
| `rag.generate_answer` | Generación OpenAI opcional o respuesta determinística. |
| `rag.langgraph.ask` | Ejecución completa del motor LangGraph. |
| `rag.langgraph.route_question` | Nodo de decisión de ruta en LangGraph. |
| `rag.langgraph.retrieve_structured` | Nodo de recuperación estructurada en LangGraph. |
| `rag.langgraph.retrieve_semantic` | Nodo de recuperación semántica en LangGraph. |
| `rag.langgraph.web_search` | Nodo de web search opcional en LangGraph. |
| `rag.langgraph.generate_answer` | Nodo final de generación/fallback en LangGraph. |

Los atributos útiles incluyen:

| Atributo | Propósito |
|----------|-----------|
| `rag.role` | Rol del solicitante usado para control de acceso. |
| `rag.has_customer_scope` | Indica si el solicitante está limitado a un cliente. |
| `rag.engine` | Motor usado cuando aplica, por ejemplo `langgraph`. |
| `rag.route` | Ruta seleccionada: `structured`, `semantic`, `combined` o `web_search`. |
| `rag.source_count` | Número de fuentes recuperadas para la respuesta. |
| `rag.certificate_count` | Número de certificados distintos representados por las fuentes. |
| `rag.structured_source_count` | Conteo de fuentes estructuradas en flujo combinado. |
| `rag.semantic_source_count` | Conteo de fuentes semánticas en flujo combinado. |
| `rag.web_result_count` | Conteo de resultados web/fallback procesados. |
| `rag.used_fallback` | Indica si la respuesta final usó el texto determinístico en vez de OpenAI. |
| `rag.question_length` | Solo longitud de la pregunta; la pregunta completa no se traza intencionalmente. |
| `rag.duration_ms` | Tiempo transcurrido aproximado del paso instrumentado. |

## Valores seguros por defecto

- `PHOENIX_ENABLED=false` por defecto.
- El modo mock sigue funcionando sin Phoenix, OpenTelemetry ni servidor Phoenix.
- Las preguntas completas, secretos, DSN, tokens y rutas de almacenamiento no se exportan como atributos de traza.
- Los identificadores de fuente y códigos de certificado no se exportan como atributos de traza; se usa telemetría basada en conteos.
- Los spans determinísticos y `rag.langgraph.*` comparten el mismo criterio: longitudes, conteos, ruta, rol y latencias; nunca payloads completos.
- Si fallan los paquetes opcionales de tracing, la configuración de Phoenix o las operaciones de ciclo de vida de spans, el tracing degrada a no-op en lugar de fallar solicitudes de la aplicación.

## Alternativa de inicio desde Python

Si el estilo de instalación de Phoenix lo permite, también se puede iniciar la aplicación desde Python:

```python
import phoenix as px

px.launch_app()
```

Mantén la API configurada con `PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006/v1/traces`.
