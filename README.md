# IT Operations Agent

A conversational IT operations prototype that combines:

- OpenAI Agents SDK for direct conversational operations
- MCP for infrastructure and ITSM integrations
- RAG-backed procedures stored in the ITSM knowledge base
- LangGraph for deterministic, stateful procedure execution

The core design principle is:

> **Use the LLM to understand intent and ambiguity. Use deterministic code to control execution.**

The system supports two distinct interaction modes:

```text
Normal chat     → agentic / ReAct-style execution
/procedure ...  → controlled procedure execution
```

---

## Architecture

```text
                         ┌─────────────────────┐
Normal chat ───────────▶│   OperationsAgent   │
                         │ OpenAI Agents SDK   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                                   MCP


/procedure
     │
     ▼
┌───────────────┐
│ ITSM KB / RAG │
└───────┬───────┘
        │ Markdown
        ▼
┌──────────────────────┐
│ Procedure Compiler   │
│ LLM                  │
└──────────┬───────────┘
           │
           ▼
   ProcedureDefinition
           │
           ▼
┌──────────────────────┐
│ Validation           │
│ deterministic        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Tool Binder          │
│ deterministic first  │
│ LLM only if needed   │
└──────────┬───────────┘
           │
           ▼
 BoundProcedureDefinition
           │
           ▼
┌──────────────────────┐
│ LangGraph Runtime    │
│ deterministic        │
└──────────┬───────────┘
           │
           ▼
          MCP
```

---

## Execution modes

### 1. Direct operations

Normal user messages are handled by the `OperationsAgent`.

Example:

```text
How many pods are running in openshift-ingress?
```

Flow:

```text
User
  ↓
OperationsAgent
  ↓
LLM selects MCP tool
  ↓
MCP call
  ↓
tool result
  ↓
LLM response
```

This path is intentionally agentic.

The model can:

- understand the request
- choose tools
- retry after a recoverable tool error
- combine multiple tool calls
- synthesize the final answer

This is useful for ad-hoc operational questions.

---

### 2. Procedure execution

Procedures are explicitly started with:

```text
/procedure <request>
```

Example:

```text
/procedure validate application state
```

The `/procedure` prefix is handled deterministically.

We do not use an LLM to decide whether a message should execute a procedure.

Flow:

```text
/procedure request
        ↓
search ITSM KB
        ↓
retrieve Markdown procedure
        ↓
compile into semantic structure
        ↓
validate
        ↓
bind steps to MCP tools
        ↓
validate complete executable plan
        ↓
LangGraph runtime
        ↓
execute steps
```

No operational step is executed before the complete plan is valid.

---

# Where we use the LLM

The LLM is used only where semantic interpretation is useful.

## Procedure compilation

The KB contains human-readable Markdown.

Example:

```markdown
### List the pods

Retrieve the pods running in **Namespace**.
```

The compiler may transform that into:

```text
system   = openshift
action   = list
resource = pods
input    = namespace
```

This is a semantic representation.

It describes **what the procedure means**.

It does not select an MCP tool.

---

## Tool binding ambiguity

A semantic step might be:

```text
openshift / list / pods
```

while the MCP server exposes:

```text
pods_list
pods_list_in_namespace
resources_list
pods_get
```

The binder first reduces candidates deterministically.

For example:

```text
200 MCP tools
    ↓
OpenShift tools only
    ↓
14 candidates
    ↓
schema accepts namespace
    ↓
3 candidates
```

Only if multiple reasonable candidates remain do we use an LLM to select between them.

The result is then validated deterministically.

The LLM cannot invent a tool.

---

## User input extraction

If a procedure is waiting for:

```text
namespace
application_name
```

and the user replies:

```text
namespace is payments and the app is checkout
```

an LLM may extract:

```text
namespace = payments
application_name = checkout
```

The output is constrained to the fields currently expected by the procedure.

For simple single-value answers, no LLM is needed.

---

# Where we do not use the LLM

The LLM is deliberately excluded from execution control.

It does not decide:

- whether `/procedure` mode is active
- whether required inputs are missing
- whether the procedure schema is valid
- whether references between steps are valid
- whether a runtime condition evaluates to true
- which step runs next
- whether a step may be skipped
- whether execution can continue after a tool failure
- whether a write operation is approved
- whether a bound tool may be replaced during execution

These decisions are implemented with normal deterministic code.

---

# Procedure format

Procedures live in the ITSM knowledge base as human-readable Markdown.

Example:

```markdown
# Validate namespace application state

## Procedure

**ID:** validate-namespace-application-state
**Version:** 1
**Risk:** low
**Confirmation required:** no

## Required information

- **Namespace** — required — Namespace to inspect.
- **Application name** — required — Application to inspect.

## Steps

### 1. Verify that the namespace exists

Check that **Namespace** exists.

If it does not exist, stop the procedure.

### 2. List the pods

Retrieve the pods running in **Namespace**.

### 3. Inspect the application

Retrieve **Application name** from **Namespace**.

## Success

The namespace exists and the application can be retrieved.
```

The KB does not contain:

```text
MCP tool names
MCP server implementation details
internal capability IDs
```

This is intentional.

The KB describes operational knowledge, not application internals.

---

# Procedure compilation

The compiler converts Markdown into a semantic internal model.

Conceptually:

```text
Markdown
   ↓
LLM
   ↓
ProcedureDefinition
```

A procedure step may look like:

```text
id       = list-pods
system   = openshift
action   = list
resource = pods
arguments:
  namespace ← procedure input "namespace"
```

The compiler may interpret the language, but it is not trusted directly.

Its output must pass deterministic validation.

---

# Deterministic validation

Before binding or execution, the application validates the compiled procedure.

Examples:

```text
procedure ID exists
version is valid
inputs are unique
step IDs are unique
input references exist
step references are valid
conditions use supported operators
future-step references are rejected
```

If the procedure cannot be represented safely:

```text
execution stops
```

There is no fallback to executing the raw Markdown.

---

# Tool binding

The semantic `ProcedureDefinition` does not know about MCP tools.

The binder creates an environment-specific executable plan.

```text
ProcedureDefinition
       ↓
current MCP tool catalog
       ↓
binding
       ↓
BoundProcedureDefinition
```

Example:

```text
Semantic step:
  openshift / list / pods

Resolved tool:
  mcp_openshift__pods_list_in_namespace

Argument mapping:
  namespace → namespace
```

This keeps the KB independent from the current MCP implementation.

If the MCP server changes its tool naming, the procedure itself does not need to change.

---

# Bound plan

Before execution starts, every executable step must be resolved.

```text
step 1 → tool A ✓
step 2 → tool B ✓
step 3 → unresolved ✗
```

In this example the procedure does not execute.

Binding is fail-closed.

Once execution starts, the bound plan is immutable.

There is no:

```text
tool rediscovery
runtime rebinding
dynamic replanning
LLM tool selection
```

---

# Why LangGraph?

LangGraph is not used because we need another agent framework.

It is used because procedure execution is a **state machine**, not an open-ended conversation.

A procedure needs explicit state such as:

```text
current step
collected inputs
step results
waiting for user
waiting for confirmation
completed
failed
cancelled
```

It also needs reliable pause/resume behavior.

That is where LangGraph fits.

---

# How we use LangGraph

We keep the graph deliberately small.

The graph is generic; procedure steps are data.

Conceptually:

```text
START
  ↓
collect_inputs
  ↓
confirm_if_required
  ↓
execute_next_step
  ↓
more steps?
  ├── yes ─────────────┐
  │                    │
  └── no → complete    │
                       │
        ◀─────────────┘
```

We do not generate one custom graph per KB procedure.

The same runtime executes different bound procedures.

---

## Pause and resume

If information is missing:

```text
status = WAITING_FOR_INPUT
```

LangGraph interrupts execution.

Example:

```text
I need:

- Namespace
- Application name
```

When the user replies, we resume the same procedure run.

```text
same procedure_run_id
same LangGraph thread
same execution state
```

The procedure does not restart.

---

## Confirmation

If the procedure requires confirmation, or the bound plan contains risky operations:

```text
status = WAITING_FOR_CONFIRMATION
```

The graph pauses before executing the relevant operation.

Example:

```text
This procedure will perform changes.

Planned operations:
- Create user
- Add user to team

Proceed?
```

Only an explicit approval resumes execution.

---

# Procedure state vs chat state

These are intentionally separate.

## Chat state

Handled by the OpenAI Agents SDK session.

It represents the conversation.

```text
user
assistant
user
assistant
```

## Procedure state

Handled by LangGraph.

It represents execution.

```text
procedure_run_id
procedure_id
current_step
inputs
step_results
status
```

Conversation history is never used as the source of truth for procedure execution.

---

# Runtime conditions

Runtime conditions use a closed internal model.

For example:

```text
exists
not_exists
equals
not_equals
greater_than
greater_than_or_equal
less_than
less_than_or_equal
is_true
is_false
is_empty
is_not_empty
```

Conditions are evaluated by deterministic Python code.

The runtime never asks:

```text
LLM, do you think this condition is true?
```

We also never execute arbitrary expressions from the KB.

No `eval()`.

---

# Failure behavior

Direct chat and procedure execution intentionally behave differently.

## Direct agent

A failed tool call may be recoverable:

```text
tool call fails
    ↓
agent corrects arguments
    ↓
second tool call succeeds
    ↓
answer
```

That is expected agentic behavior.

## Procedure runtime

A bound step failure stops the procedure:

```text
step 1 ✓
step 2 ✗
step 3 not executed

status = FAILED
```

The runtime does not dynamically choose another tool.

This preserves the meaning of the validated execution plan.

---

# End-to-end procedure flow

A complete procedure interaction looks like this:

```text
User
 |
 | /procedure validate application
 v
Procedure router
 |
 v
ITSM KB / RAG
 |
 | Markdown
 v
Procedure Compiler (LLM)
 |
 | semantic plan
 v
Deterministic Validator
 |
 v
Tool Binder
 | \
 |  \ MCP catalog
 |   \
 v
Bound Procedure
 |
 v
LangGraph Runtime
 |
 | missing input?
 +------ yes ------> interrupt
 |                    |
 |                    v
 |                  User
 |                    |
 |<------ resume -----+
 |
 | confirmation required?
 +------ yes ------> interrupt
 |                    |
 |                    v
 |                  User
 |                    |
 |<------ resume -----+
 |
 v
Execute bound MCP step
 |
 v
Store result
 |
 | more steps?
 +------ yes ------> next step
 |
 no
 |
 v
COMPLETED
```

---

# Deterministic boundary

The most important boundary in the project is:

```text
Human language
      ↓
LLM interpretation
      ↓
validated structure
----------------------------- execution boundary
      ↓
deterministic runtime
      ↓
MCP
```

The LLM is allowed to help before the execution boundary.

Once execution begins, it cannot modify the plan.

---

# Design principles

```text
LLM
→ understand language and ambiguity

Python
→ validate and enforce rules

Tool Binder
→ resolve semantic intent to current MCP capabilities

LangGraph
→ control execution state and pause/resume

MCP
→ perform actual operations
```

Or, in one sentence:

> **LLM for understanding, deterministic code for control, LangGraph for execution state, MCP for actions.**

---

# Intentionally out of scope

The first version does not implement:

- multi-agent orchestration
- dynamic replanning
- automatic rollback
- arbitrary retries
- loops
- parallel procedure steps
- nested procedures
- arbitrary code from the KB
- generic workflow DSLs
- tool selection during procedure execution

The goal is to keep procedure execution predictable, observable, and safe.

---

# Recommended validation scenarios

A small initial evaluation set should cover:

```text
1. Read-only procedure
2. Procedure requiring user inputs
3. Write procedure requiring approval
4. MCP tool failure
5. Unsupported condition
6. Ambiguous tool binding
7. Cancellation
8. Two independent chat sessions
```

At this point, improvements should be driven by real procedure failures rather than adding more architecture.

## Implemented prototype safeguards

Conditions contain a step/result operand, one of the operators above, and either
a scalar comparison value or a declared `source_input`. Dot-separated object keys
are relative to the stored `structuredContent`. `source_text` is explanatory
provenance; it is not matched against Markdown or interpreted at runtime. The
LLM compiler alone interprets condition prose. Python first parses metadata,
inputs, ordered headings, normalized IDs, step bodies and success text into a
`SourceProcedure`. The model receives that fixed structure and returns only
semantic entries keyed by those IDs. Exact coverage is checked before merging
in source order into `ProcedureDefinition`; closed-model validation checks references,
operators and operands. There
are no compound expressions, array indexing, derived counts or output-to-argument
transformations. Missing paths and type mismatches fail safely. `exists` tests
for a non-null resolved value; it does not hide a missing path. A false precondition
or a true stop condition stops the run with `CONDITION_ERROR`; steps are not skipped.
Known incompatible output paths/operators prevent the run from starting.
Conditions belong to structurally verified source steps. A stop rule using an
already-known previous result is checked before the next call;
a stop rule using the current result is checked immediately after that call.

Bindings capture input/output schemas and an execution-risk classification from
explicit MCP annotations or HTTP-method metadata. Absent or contradictory metadata
is `UNKNOWN`, including unannotated tools with read-like names. `WRITE`,
`DESTRUCTIVE` and `UNKNOWN` steps require approval before their first call. Approval
covers all remaining risky steps. A procedure-level confirmation that explicitly
summarizes those changes also covers this approval. Input values are omitted from
the browser summary because the prototype has no secret-field metadata.

Each run stores a serialized bound-plan snapshot, a plan digest and a catalog
fingerprint. The existing SDK server connections refresh metadata only to check
the bound identities/schemas/risk before execution and after pauses; they do not
discover invocation handles, choose replacements or rebind. Relevant catalog
changes stop the run. Final resolved arguments are validated locally before every
call. No tool-schema defaults or other runtime values are fabricated.

Internal step audit records contain server/tool identity, status, ordered
start/completion markers, whether an operation was attempted, and a normalized
error category. Browser responses omit audit data, raw results, credentials and
internal tool names. The legacy stateless `/api/chat` preview still does not
execute; concrete binding diagnostics require `debug_mode=true` in its request.

Run the full offline suite with `make test`. Run the complete procedure evaluation
set with `.venv/bin/python -m pytest tests/test_procedure*.py tests/test_agent.py`.
The focused evaluations are in
`tests/test_procedure_evaluation.py`, with three human-readable Markdown fixtures
under `tests/fixtures/procedures`. No real providers or MCP servers are called.

Storage remains local and in memory, with one active run per conversation.
There is no durable crash recovery or exactly-once guarantee for external writes.
Cancellation lets an in-flight call finish and never undoes it. Catalog validation
is a preflight check, not a lock on a remote server; later unavailability fails the
bound call without substitution. Risk annotations are environmental metadata,
not a proof of remote tool implementation behavior. Textual success criteria are
not independently evaluated; completion reports actual step outcomes only.
