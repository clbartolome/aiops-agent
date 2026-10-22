# Operations Agent

Install with `make install`, copy `.env.example` to `.env` and configure the model
and MCP endpoints, then start the terminal web UI with `make run-app`.
Open http://127.0.0.1:8000. Run tests with `make test`.

The UI supports multiple independent conversations. Use **+ New** to create one
and the conversation sidebar to switch between them. Clarification replies continue
the selected conversation using OpenAI Agents SDK `SQLiteSession`, passed directly
to `Runner.run`. The SDK owns conversation history, including MCP tool exchanges;
only user and assistant text is rendered in the browser.

Each UUID conversation uses its own in-memory SQLite session. Minimal titles and
timestamps are held in memory, with a per-conversation lock to serialize turns.
This storage is intended for local development in a single application process;
conversations disappear when the application restarts. No authentication or per-user
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
