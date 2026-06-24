# Configuración de integración real local

El repositorio corre en **modo mock por defecto**. El modo real es opcional y solo debe usarlo un operador local que configure sus propias credenciales de OpenAI, MySQL de solo lectura y S3 con mínimo privilegio en `~/.config/certilab-agentic-rag/.env` o en la ruta indicada por `CERTILAB_RAG_ENV_FILE`.

## Ruta rápida

1. Instala las dependencias de desarrollo e integraciones futuras con `uv`:

   ```bash
   uv sync --extra dev --extra future-integrations
   ```

2. Copia el archivo de ejemplo al archivo local externo de operador:

   ```bash
   mkdir -p ~/.config/certilab-agentic-rag
   cp .env.example ~/.config/certilab-agentic-rag/.env
   ```

3. Mantén el modo mock mientras desarrollas los ejercicios del curso:

   ```env
   APP_MODE=mock
   ```

4. Para experimentos reales locales, configura `APP_MODE=real` en ese archivo externo y completa solo tus propias credenciales locales. No copies secretos en Git, documentación, capturas, issues ni conversaciones.

Si necesitás usar otra ubicación, exportá `CERTILAB_RAG_ENV_FILE=/ruta/operator.env`. Las pruebas usan `CERTILAB_RAG_DISABLE_DOTENV=true` para no cargar `.env` del proyecto ni archivos externos.

## Mapeo de variables de entorno

| Propósito | Variable RAG preferida | Alias compatibles con Laravel |
|---|---|---|
| Modo de aplicación | `APP_MODE=mock\|real` | — |
| API key de OpenAI | `OPENAI_API_KEY` | — |
| Modelo de embeddings | `OPENAI_EMBEDDING_MODEL` | — |
| Modelo de chat | `OPENAI_CHAT_MODEL` | — |
| URL de Qdrant | `QDRANT_URL` | — |
| Colección Qdrant | `QDRANT_COLLECTION` (default: `certilab-rag`) | — |
| API key de Qdrant | `QDRANT_API_KEY` (cloud) | — |
| Proveedor de embeddings | `EMBEDDING_PROVIDER` (`auto\|openai\|local`) | — |
| Modelo local de embeddings | `SENTENCE_TRANSFORMERS_MODEL` (default: `all-MiniLM-L6-v2`) | — |
| DSN de MySQL | `MYSQL_READONLY_DSN` reservado para soporte futuro | — |
| Host/puerto/base/usuario/password de MySQL | — | `DB_HOST`, `DB_PORT`, `DB_DATABASE`, `DB_USERNAME`, `DB_PASSWORD` |
| Bucket S3 | `S3_BUCKET_NAME` | `AWS_BUCKET` |
| Región AWS | `AWS_REGION` | `AWS_DEFAULT_REGION` |
| Prefijo S3 | — | `AWS_STORAGE_PREFIX` |
| Nombre del disco S3 | — | `CERTIFICATES_STORAGE_DISK` |
| Credenciales AWS | — | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` |

Los nombres RAG preferidos mantienen este servicio desacoplado de Laravel. En la versión actual, el conector MySQL real usa los campos Laravel compatibles `DB_HOST`, `DB_PORT`, `DB_DATABASE`, `DB_USERNAME` y `DB_PASSWORD`; `MYSQL_READONLY_DSN` queda documentado como variable reservada para una futura ruta DSN segura. Los alias compatibles con Laravel se aceptan para que un operador local pueda reflejar la forma del entorno existente sin renombrar todo.

## Modo mock

El modo mock no requiere servicios externos:

```env
APP_MODE=mock
OPENAI_API_KEY=
MYSQL_READONLY_DSN=
S3_BUCKET_NAME=
```

La API usa fixtures JSON anonimizados y fixtures locales de texto de PDF bajo `data/`.

## Plan de modo real

El modo real está diseñado alrededor de tres componentes:

1. **Metadatos de MySQL** — leer metadatos de certificados, clientes e historiales mediante un usuario de solo lectura. El conector usa consultas `SELECT` con allowlist y excluye campos sensibles como passwords, passwords en texto plano, remember tokens, ruc, email y phone.
2. **Qdrant como vector store** — los chunks se embeben y almacenan en Qdrant con filtrado por tenant obligatorio. Cada punto incluye `customer_id` y `code` en el payload.
3. **PDFs en S3** — resolver claves de PDF de certificados bajo un prefijo permitido, rechazar path traversal y solo después generar una URL prefirmada o traer bytes para futura extracción de texto (scope-defer).

### Qdrant con Docker Compose

```bash
docker compose up -d qdrant
```

Qdrant expone REST en `http://localhost:6333` y gRPC en `localhost:6334`. La colección se crea idempotentemente en el primer arranque del pipeline en modo real.

### Proveedor de embeddings

La selección del proveedor se controla con `EMBEDDING_PROVIDER`:

| Valor | Comportamiento |
|---|---|
| `auto` (default) | OpenAI si hay `OPENAI_API_KEY`; si no, sentence-transformers local |
| `openai` | OpenAI `text-embedding-3-small` (fallback a local si falta la key) |
| `local` | sentence-transformers offline (sin red ni credenciales) |

Si ambos fallan (paquete no instalado, modelo no disponible), el proveedor retorna un vector cero determinístico para que el pipeline degrade sin caer.

Instalar dependencias locales:

```bash
uv sync --extra dev --extra future-integrations --extra local-embeddings
```

### Allowlist de columnas y exclusión de PII

El conector MySQL selecciona solo columnas explícitamente permitidas:

| Tabla | Columnas permitidas |
|---|---|
| `certificates` | id, code, case_number, request_number, document_type, issue_date, service_date, customer_id, user_id, pdf_document_path, qr_code, status |
| `customers` | id, company_name |
| `certification_histories` | id, certification_id, user_id, action, ip, user_agent, created_at |

**PII nunca leído, embebido, indexado ni expuesto**: password, plain_password, remember_token, ruc, email, phone. La tabla `users` nunca se consulta.

### Filtrado por tenant

Cada chunk almacenado en Qdrant incluye `customer_id` en su payload. El `SemanticRetriever` computa el conjunto de IDs permitidos a partir del `AccessScope` del caller y lo pasa como filtro a la búsqueda. Un cliente nunca ve resultados de otro cliente.

OpenAI sigue siendo opcional hasta que se invoque el camino real de embeddings o generación de respuesta. Los modelos por defecto son placeholders para indexación y respuestas reales futuras:

```env
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini
```

## Checklist de seguridad

- [ ] El archivo de operador `~/.config/certilab-agentic-rag/.env` o `CERTILAB_RAG_ENV_FILE` queda fuera del repositorio y no se imprime ni se comparte.
- [ ] Las credenciales de MySQL pertenecen a un usuario de **solo lectura** con acceso únicamente a las tablas y columnas permitidas para la ingesta.
- [ ] Las credenciales de S3 tienen acceso de mínimo privilegio al bucket de certificados y solo al `AWS_STORAGE_PREFIX` configurado.
- [ ] No se copian valores secretos desde Laravel hacia este repositorio, documentación, commits, tickets ni salida de terminal.
- [ ] La ingesta real se prueba localmente sin commitear artefactos generados que contengan datos productivos.

## Verificación

Ejecuta la suite determinística sin servicios reales:

```bash
uv run pytest
uv run ruff check .
uv run mypy app
```

Estos comandos deben pasar en modo mock sin OpenAI, MySQL, S3, `pymysql` ni `boto3` instalados.
