# Contrato de datos

El proyecto soporta dos modos: **mock** (fixtures JSON locales) y **real** (MySQL de pruebas). Los modelos canónicos de dominio son los mismos en ambos modos; el loader de cada modo es responsable de mapear los datos de origen a esos modelos.

## Modelos canónicos de dominio

Estos son los campos que el pipeline RAG usa internamente, independientemente del modo activo.

### Customer

| Campo | Tipo | Descripción |
|---|---|---|
| `id` | integer | Identificador del cliente. |
| `name` | string | Nombre del cliente (`company_name` en la BD real). |
| `segment` | string | Segmento general (solo en fixtures mock; no existe en la BD real). |

### Certificate

| Campo | Tipo | Descripción |
|---|---|---|
| `id` | integer | Identificador interno. |
| `code` | string | Código público del certificado. |
| `customer_id` | integer | Dueño del certificado; obligatorio para aislamiento de tenant. |
| `status` | string | Estado operativo. |
| `emitted_at` | date | Fecha de emisión (`issue_date` en la BD real). |
| `technician_id` | integer | Identificador del técnico (mock) / `user_id` (BD real). |
| `equipment` | string | Equipo certificado (solo en fixtures mock; no existe en la BD real). |
| `pdf_path` | string | Ruta del texto PDF (`pdf_document_path` en la BD real). |
| `case_number` | string \| None | Número de expediente (solo en modo real). |
| `request_number` | string \| None | Número de solicitud (solo en modo real). |
| `document_type` | string \| None | Tipo de documento (solo en modo real). |
| `service_date` | date \| None | Fecha de servicio (solo en modo real). |
| `qr_code` | string \| None | Código QR del certificado (solo en modo real). |

`pdf_path` es un dato interno del pipeline. Las respuestas públicas usan `source_id` sanitizado, no rutas de almacenamiento.

### CertificateHistory

| Campo | Tipo | Descripción |
|---|---|---|
| `id` | integer | Identificador del evento. |
| `certificate_id` | integer | Certificado asociado (`certification_id` en la BD real). |
| `event` | string | Evento operativo (`action` en la BD real). |
| `occurred_at` | date | Fecha del evento (`created_at` en la BD real). |
| `note` | string | Nota (solo en fixtures mock; no existe en la BD real). |

## Mapeo real → canónico (MySQLLoader)

| BD real | Modelo canónico | Tabla |
|---|---|---|
| `customers.company_name` | `Customer.name` | customers |
| `certificates.issue_date` | `Certificate.emitted_at` | certificates |
| `certificates.pdf_document_path` | `Certificate.pdf_path` | certificates |
| `certificates.user_id` | `Certificate.technician_id` | certificates |
| `certification_histories.certification_id` | `CertificateHistory.certificate_id` | certification_histories |
| `certification_histories.action` | `CertificateHistory.event` | certification_histories |
| `certification_histories.created_at` | `CertificateHistory.occurred_at` | certification_histories |

## Columnas permitidas en la BD real (allowlist)

Solo estas columnas se leen. Nada fuera de esta lista puede entrar al pipeline.

| Tabla | Columnas permitidas |
|---|---|
| `certificates` | id, code, case_number, request_number, document_type, issue_date, service_date, customer_id, user_id, pdf_document_path, qr_code, status |
| `customers` | id, company_name |
| `certification_histories` | id, certification_id, user_id, action, ip, user_agent, created_at |

## Campos prohibidos (PII — nunca se leen, embeben ni exponen)

| Campo | Tabla de origen | Razón |
|---|---|---|
| `password` | users, certificates | Credencial |
| `plain_password` | users | Credencial en texto plano |
| `remember_token` | users | Token de sesión |
| `ruc` | customers | Identificador fiscal |
| `email` | customers, users | Dato personal |
| `phone` | customers | Dato personal |

La tabla `users` no se consulta en ninguna ruta del pipeline. Los campos anteriores también están excluidos de embeddings, snippets y logs de tracing.

Además quedan excluidos: variables `.env`, credenciales de AWS, DSN de base de datos, API keys, PDFs reales sin anonimización y cualquier campo no declarado en la allowlist.

## Reglas para integración real

- Todo registro debe llevar `customer_id` o un equivalente verificable.
- Las consultas MySQL usan un usuario de **solo lectura** y la allowlist de columnas del conector.
- Qdrant almacena `customer_id` como payload filtrable obligatorio en cada punto.
- El `SemanticRetriever` filtra por los IDs de chunk permitidos según el `AccessScope` del caller antes de consultar Qdrant.
- Los objetos S3 se resuelven solo después de autorizar el `customer_id` correspondiente.
