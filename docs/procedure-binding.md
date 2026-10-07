Procedure tool binding (Step 3)
===============================

The compiler and semantic `ProcedureDefinition` remain unchanged. Binding lives
in `app/procedure_binding.py`; environment-specific models live in
`app/procedure_binding_models.py`:

- `AvailableTool`: SDK-origin server, public namespaced tool name, description,
  current input/output schemas, explicit annotations and operation metadata.
  It has no invocation function.
- `ToolArgumentBinding`: semantic step argument name and concrete parameter name.
- `BoundProcedureStep`: semantic step ID, server/tool names, argument mappings,
  captured schemas and `READ` / `WRITE` / `DESTRUCTIVE` / `UNKNOWN` classification.
- `BoundProcedureDefinition`: original semantic definition plus a complete ordered
  set of bindings and a catalog fingerprint. Partial, duplicate, conflicting,
  or missing bindings are rejected. Bound model fields are frozen; the runtime
  stores its own serialized snapshot and checks its plan digest.

`app/mcp.py` adapts metadata from the existing SDK `get_all_tools` discovery with
`include_server_in_tool_names=True`. Namespacing and provenance come from the SDK;
the application does not regenerate public names or maintain a tool list. The
procedure path keeps its ITSM retrieval connection alive and opens only additional
systems needed by semantic operations. Existing SDK tool-list caching is reused.
Direct OperationsAgent behavior, transports, authentication, compiler, and
session storage are unchanged.

Resolution
----------

1. Filter by the semantic system's stable server name. Missing systems fail;
   there is no cross-server guessing.
2. Filter known primary-resource mismatches before looking at input parameters.
   Resource-first/verb-first tool names and unambiguous descriptive direct objects
   identify lightweight normalized resources. Obvious plural forms normalize to
   singular. Namespace scope parameters never establish primary resource. Opaque
   or generic resource metadata remains unknown and requires semantic selection.
3. Filter input schemas by possible one-to-one mappings of all semantic step
   arguments. Prefer exact normalized parameter names. Small aliases support
   `namespace`/`namespace_name`, `username`/`user_name`, and qualified name/ID
   parameters such as `application_name` → `name`. There is no general fuzzy
   matcher. Every required concrete parameter must be covered.
4. Rank schema-compatible tools using clear lexical action/resource evidence.
   Strong matches may narrow candidates; if there are none, retain all schema
   candidates for semantic selection. Verbs in procedure intent need not match
   concrete tool naming conventions. A single clear action/resource match with
   one valid mapping binds deterministically. A questionable singleton still
   requires selection. Otherwise, a dedicated structured model receives only
   the step, candidate servers/names/descriptions, schemas, and allowed mappings.
   More than 12 candidates fails as
   unresolved ambiguity rather than sending a large catalog. Refusal is represented
   by null `tool_name` or `ambiguous=true`; no arbitrary first-match fallback is
   used. Equivalent descriptions and compatible argument schemas also fail closed,
   even if the selection model picks one.
5. Independently validate existence, server provenance, primary-resource compatibility, candidate membership,
   unique mappings, known parameter names, required coverage, parameter concepts,
   and schema compatibility. All steps must succeed before any bound result is
   returned. Logs distinguish server/resource/schema counts, strong purpose matches,
   semantic candidates and selection use, plus the validated selected tool and
   argument mapping. Small per-stage candidate-name lists and resource/required-
   parameter/short-description summaries appear at DEBUG level.
   Failures are logged with procedure/step IDs;
   raw model payloads are omitted and known credentials are redacted.

Scope and limits
----------------

The current mapping supports flat object argument schemas with scalar or nullable
scalar parameters. Conditional/composed top-level objects, references, nested
object/array parameters, and dynamic parameter schemas fail closed. JSON Schema
validity and any declared procedure defaults are checked using `jsonschema`.
A required tool parameter cannot depend on an optional input with no default.
No default or actual user value is written into a bound step. Because semantic
inputs have no declared type, compatibility checks cannot prove the validity of
future user values; execution must validate those values against the tool schema.
This phase selects only compatible parameter names and checks known defaults.

Closed preconditions and stop conditions remain in the semantic definition;
the binder never evaluates them. The runtime validates known output paths and
evaluates only structured results. Success text is provenance, not an executable
assertion. Every current semantic
step carries an action/resource and requires a binding. A standalone logical
verification without a tool that implements that operation remains unresolved;
it is never silently skipped or substituted with an unrelated read operation.
Binding does not execute tools, run conditions, approve operations, or retry them.

The session UI continues into the LangGraph runtime after binding and retrieval
connections finish cleanup. The legacy stateless `/api/chat` endpoint previews
semantic operations without execution. Concrete names and mappings in that preview
require the explicit request field `debug_mode=true`.
Failures retain a generic browser message and do not store partial bindings in
session history. Only the existing KB search and article retrieval tools are
invoked during retrieval/binding; discovery uses metadata-only tool listing.

Before execution (and again after a pause), existing SDK servers refresh metadata
solely to verify frozen bound identities, schemas and risk classifications. No
FunctionTool discovery, model selection or rebinding occurs. Missing tools and
fingerprint changes stop the run. Unrelated catalog entries do not affect the
fingerprint. Required-list/enum ordering and object property order are normalized.
Explicit read-only annotations or HTTP-method metadata can establish `READ`;
missing or contradictory signals remain `UNKNOWN`, which requires approval.
