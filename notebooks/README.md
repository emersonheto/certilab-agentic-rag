# Notebooks

Este directorio contiene notebooks de exploración académica controlada. El notebook principal de entrega es [`certilab_agentic_rag_demo.ipynb`](certilab_agentic_rag_demo.ipynb).

## Notebook de entrega

`certilab_agentic_rag_demo.ipynb` muestra un recorrido offline y mock-only del flujo Agentic RAG:

Referencia académica de la consigna: **Building an Adaptive RAG System with LangGraph, OpenAI and Tavily** — https://levelup.gitconnected.com/building-an-adaptive-rag-system-with-langgraph-openai-and-tavily-c4ee39d2f021

- request de ejemplo a `/ask` con `X-Demo-Token` placeholder;
- respuestas determinísticas para rutas `structured`, `semantic` y `combined`;
- fuentes sanitizadas con `source_id`, sin rutas internas ni datos productivos.

Reglas de uso:

- Usar únicamente datos mock o anonimizados.
- No guardar credenciales, secretos, tokens, `.env`, PDFs reales ni exports de producción.
- No ejecutar llamadas de red ni proveedores externos desde notebooks de entrega.
- Convertir experimentos útiles en código dentro de `app/` antes de considerarlos parte del MVP.
