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

## Procedure mode

Typing `/procedure <request>` in any conversation (same chat UI, same endpoints)
looks up a matching knowledge-base article in the configured ITSM MCP server,
deterministically parses its `## Procedure` Markdown section, and runs it step by
step through a LangGraph-controlled runtime: collecting any required inputs,
asking for confirmation when the article requires it, executing each step with a
bounded Agents SDK + MCP tool loop, and pausing for explicit approval before any
non-read-only tool call. `/cancel` stops the active run for that conversation at
any point. Normal chat messages are unaffected and keep using the existing
`OperationsAgent` path; see `app/procedure/__init__.py` for the full routing logic.

Three small, representative example procedures live under `fixtures/procedures/`
and are used by the automated test suite (`tests/test_procedure_e2e.py`):
read-only inspection (`inspect_namespace_health.md`), multiple required inputs
(`check_application_pods.md`), and a confirmed write operation
(`launch_aap_job.md`). All automated tests mock the model and the MCP transport;
none of them ever call a real MCP server or model API.

### Manual smoke test against real MCP servers

Once the automated suite passes, you can validate the same flow against your
*actual* configured MCP servers (never done in `make test`). Load a real ITSM
knowledge base article whose content matches one of the `fixtures/procedures/`
examples (or write your own following the same Markdown convention), configure
`.env` with real MCP endpoints/credentials, run `make run-app`, and in the chat UI:

```
/procedure inspect namespace health
```

Expected interaction (exact field/step names depend on your real KB article):

1. The assistant asks for the missing required input(s), e.g. `- Namespace`.
2. Reply with the value, e.g. `namespace=payments` (or just `payments` if it is
   the only missing field).
3. The assistant runs each step in order against your real MCP tools and
   reports a final `Procedure completed successfully.` / `Procedure stopped as
   instructed.` / `Procedure failed.` message with a per-step checklist.
4. For a procedure with `Confirmation required: yes`, reply `yes`/`no` to the
   `Proceed?` prompt before any step runs.
5. For any step that proposes a non-read-only tool call, reply `yes`/`no` to
   the `PROCEDURE · APPROVAL REQUIRED` prompt before that tool is ever invoked.

Watch the application logs (`procedure_run_id=`, `procedure_id=`, `step_id=`,
`status=` fields) to confirm the real MCP tool calls and their outcomes.
