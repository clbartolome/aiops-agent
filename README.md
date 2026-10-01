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
MCP tools, model settings, and conversation behavior are unchanged.

For a diagnostic-only tool comparison, load `.env` as for `make run-app` and run
`python -m scripts.diagnose_model_tools --case mcp-only`. Other useful cases are
`all`, `single-mcp`, `single-strict-mcp`, and `all-strict-local` (reproduces the old
failure). This command logs only tool counts, names, origins, and schema shapes;
it does not install a tool filter in the application. Its default request is a
read-only pod count; `--message` can supply another diagnostic request.
