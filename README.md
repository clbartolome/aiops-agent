# Operations Agent

Install with `make install`, copy `.env.example` to `.env` and configure the model
and MCP endpoints, then start the terminal web UI with `make run-app`.
Open http://127.0.0.1:8000. Run tests with `make test`.

The UI supports multiple independent conversations. Use **+ New** to create one
and the conversation sidebar to switch between them. Clarification replies continue
the selected conversation using OpenAI Agents SDK `SQLiteSession`, passed directly
to `Runner.run`. The SDK owns conversation history, including MCP tool exchanges;
only user and assistant text is rendered in the browser.

Conversation metadata and SDK-owned messages persist in SQLite at
`SESSION_DB_PATH` (default `data/sessions.sqlite3`), with a per-session lock to
serialize turns. This storage is intended for a single application process;
active procedure runtime state is still held in memory. No authentication or per-user
isolation is provided yet. The original `/api/chat` endpoint remains available for
independent requests; the UI uses `/api/sessions` and session-specific messages.

The configured `gpt-oss-20b` provider rejects strict function-tool schemas with
`structure_info not used with HarmonyParser`. The local clarification tool uses
SDK `strict_mode=False`; its required arguments are still validated by the SDK.
MCP tool schemas, model settings, and conversation behavior are preserved.

For a diagnostic-only tool comparison, load `.env` as for `make run-app` and run
`python -m scripts.diagnose_model_tools --case mcp-only`. Other useful cases are
`all`, `single-mcp`, `single-strict-mcp`, and `all-strict-local` (reproduces the old
failure). This command logs only tool counts, names, origins, and schema shapes;
it does not install a tool filter in the application. Its default request is a
read-only pod count; `--message` can supply another diagnostic request.

MCP servers still advertise all tools, and the application discovers the full
namespaced tool list. Before either OperationsAgent or the Procedure Step Executor
runs, the application exposes only tools allowed by `OPENSHIFT_MCP_ALLOWED_TOOLS`,
`AAP_MCP_ALLOWED_TOOLS`, and `ITSM_MCP_ALLOWED_TOOLS`. Use comma-separated, exact
SDK names such as `mcp_openshift__pods_list_in_namespace`; whitespace and duplicate
entries are ignored. Local tools such as `request_user_input` remain separate.

Unset variables use the demo defaults in `.env.example` (8 OpenShift, 11 AAP,
10 ITSM tools). A defined but empty variable exposes zero tools for that server.
Names with the wrong server prefix are rejected; configured names missing from
discovery produce warnings and are never fabricated. Unknown servers default to
zero exposed tools. Logs report discovered, allowed, and exposed counts per server
and the combined MCP surface. Change the variables and restart to change model
visibility without modifying MCP servers. The fixed `/procedure` KB retrieval
continues to call ITSM `rag_search_kb` directly.

External incident intake uses `POST /incidents` with required `title`, `message`,
and `severity` (`critical`, `warning`, or `info`, case insensitive). Incidents
persist in SQLite at `INCIDENT_DB_PATH` (default `data/incidents.sqlite3`). The
right-side inbox polls every second; opening an item displays its details
in its own persisted chat session without calling the agent, searching the KB,
or starting a procedure. Each incident owns a session created at intake, whose
first persistent message references the incident record. Incident sessions appear
only in the right inbox; normal conversations appear on the left. Both selectors
load the same central chat and use the same session/message APIs. New intake
automatically starts background procedure preparation using the title and message;
opening an incident only displays the work already belonging to that session.
Existing incidents receive stable session IDs when first loaded after migration.

Try while `make run-app` is running:

```bash
curl -X POST http://127.0.0.1:8000/incidents \
  -H 'Content-Type: application/json' \
  -d '{"title":"High CPU on web-server-prod","message":"CPU utilization above 95% for 10 minutes on node worker-03.","severity":"critical"}'
```

Incident preparation shares the `/procedure` retrieval, deterministic parser,
input extractor, missing-input collection, and sequential executor. Retrieval uses
only the existing fixed ITSM `rag_search_kb` call. KB results, extracted parameter
labels/values, and input questions persist in the incident session. Progress never
does. Incident replies go to its active procedure context, never OperationsAgent.

Even when KB metadata says confirmation is unnecessary, incident-originated runs
stop in `WAITING_FOR_CONFIRMATION` before step one. Exact human replies `yes`,
`run it`, or `proceed` authorize the shared executor; `no`, `do not run it`,
`cancel`, or `/cancel` cancel it. Other replies leave the gate closed. Incident
text and extracted values never count as consent. The execution loop also rejects
unconfirmed incident contexts. Active incident replies return 202 and run in the
owning session task, including API callers without the Prefer header.

Incidents become `IN_PROGRESS` at preparation start, remain so while waiting for
inputs/confirmation, and become `DONE` only after successful execution. No match,
preparation errors, rejection, STOP, or FAILED return them to `NEW` for manual
review. `procedure_run_id` links the incident to its shared procedure context.

The web console submits session messages with `Prefer: respond-async`, receiving
`202 Accepted`. Application-owned asyncio tasks process each session independently;
navigation and HTTP disconnects do not cancel them. A second submission to a busy
session returns 409. Other sessions can process concurrently. Normal conversation callers
without that header retain the existing synchronous/NDJSON response contract.

Both sidebars poll lightweight session state every second: a small animated
loader means processing, and an amber dot means unread output. The selected history reloads only when its revision
changes. Opening/rendering a session acknowledges that exact revision through
`POST /api/sessions/{id}/read`; unseen newer output stays unread. Unread state is
shared by this single-user console and persists across restarts. Progress is
runtime state only and never joins chat history.

In-process work and active procedure runtime state do **not** survive a server
restart. Completed messages, sessions, incidents, and unread state remain durable.
Waiting procedure contexts are also runtime-only: after a server restart an old
confirmation message does not authorize execution, and its reply cannot resume a
lost context. Persisted messages and run linkage remain available for review.


Sidebar titles use two lines with full titles available on hover. A small delete
control appears on hover or keyboard focus (always visible on touch screens),
with a confirmation before removal. `DELETE /api/sessions/{id}` removes normal
conversation history; incident-owned sessions must be removed through
`DELETE /api/incidents/{id}`, which also clears their session and messages.
Both endpoints reject busy sessions with 409 and return 204 after deletion.
