Initial procedure compilation investigation (2026-10-05)
=======================================================

This section records the pre-refactor capability-based contract. The current
semantic contract is described below; the earlier restrictions are historical.

The retrieved article is `validate-namespace-application-state`, titled
“Validate namespace application state”. No article or compiler contract was changed.

The live extraction failed deterministic reference validation inside Pydantic:

```
steps.0.stop_condition.source_step
step=check-that-the-namespace-exists
reference=check_that_the_namespace_exists
error=Unknown step reference.
```

The underscore reference does not match the generated hyphenated ID. Even with
the ID corrected, this condition is attached to the first/current step; the
contract allows references only to earlier steps. The source stop rule must be
represented on a following action. LLM extraction can vary between requests;
this is the exact failure observed in this reproduction, not a saved diagnosis
of the earlier browser request.

Independent source incompatibilities:

- Step 3, “Verify that enough pods exist”, has no `Capability`. Every
  `ProcedureStep` requires one. A standalone logical check has no representation
  in this model. Do not invent a capability to fill the gap. The article must
  use an actual supported capability or express its rule as a precondition or
  stop condition on a subsequent action, referencing the earlier pod-list step.
- Step 4 currently depends on the result of that standalone logical step. If
  step 3 is removed, this condition must refer to the earlier actual action;
  it cannot retain a reference to a nonexistent validation step.
- The retrieved Markdown contains CRLF. The current source-section and
  capability regular expressions do not consistently accept carriage returns.
  For example, section names are captured with a trailing carriage return and
  fail the supported-section check. Author the article with LF to satisfy the
  existing parser. Diagnostics do not normalize the source or relax the parser.

Model boundaries that matter for rewriting the article:

- `StepArgument` supports only a named procedure input (`source_input`). It has
  no previous-step output path, literal value, or computed argument binding.
  The current article's action argument bindings use declared inputs and fit.
- `ProcedureCondition` contains one earlier `source_step` and a nonempty textual
  `expression`. The pod-count threshold can be preserved as prose referring to
  the pod-list action; there is no typed comparison, input operand, output path,
  or multi-source logical expression model. Do not extract such extra fields.
- `success_criteria` is an optional string copied exactly from `## Success`.
  The article's bullet list is representable as text. It is not an executable
  set of assertions, and the compiler does not evaluate it.

Diagnostics
-----------

Retrieval logs `KB retrieved` and `Procedure marker found`. Compilation logs
structured compilation start/completion, Pydantic completion, and deterministic
source-validation completion. A failed stage logs field paths and reasons.
Reference validators remain in Pydantic and retain the same acceptance rules;
their failures are explicitly categorized as deterministic validation.

Structured JSON is validated directly against the same strict
`ProcedureDefinition`, preserving Pydantic errors rather than the SDK wrapper
that may contain the entire output. Logs omit validation inputs, model output,
provider response bodies, and stack traces. The shared redactor removes configured
and environment credentials. Untrusted arbitrary transport exception payloads
remain withheld; API exceptions retain their redacted error message. Browser
responses remain generic and no procedure steps execute.


Semantic Step 2 refactor
-----------------------

The compiler now ends at a semantic `ProcedureDefinition`. A step has a required
source description, an optional logical `system`, a small `action` verb, and a
normalized `resource`; there is no capability or concrete tool field. The existing
flat model/compiler/routing modules are retained. No binding layer, execution
engine, registry, or persistence was added.

Legacy capability annotations and obvious MCP identifiers/transport are rejected,
even if extraction omits them. Articles must remove those implementation details.
The [namespace example](examples/validate-namespace-application-state.md) retains
all six operations, including the previously unrepresentable logical pod-count
verification, using prose and declared inputs only. Tests compile its semantic
extraction through the real compiler and deterministic validators with LF and
CRLF input; the model response is mocked to keep tests independent of providers.
The remote KB article has not been edited.

Current-step stop conditions are supported. Preconditions still require earlier
steps, and neither kind may reference a future step. Textual conditions and success
criteria are retained without evaluation. Argument values can only come from
procedure inputs. Step descriptions must be source quotations, and input mentions,
step titles/order, and logical system/resource mentions receive bounded source
checks. These lexical checks are not a semantic proof of the extracted action.

The original stage diagnostics and generic browser failure message are preserved.
Successful summaries show `system / action / resource`. The compiler passes no
tools or handoffs to the model and does not inspect the discovered MCP catalog.


Step 3 adds a separate binding phase after semantic compilation; the compiler
itself remains unchanged. Successful summaries now include SDK-discovered concrete
tools and argument-name mappings. See [binding details](procedure-binding.md) for
candidate filtering, complete-binding validation, and current schema limits.

Final prototype contract
------------------------

The earlier Step 2 notes above describe the historical textual condition model.
Executable conditions now require a closed `operand` / `operator` representation.
`operand.step_id` replaces `source_step`; a dot-separated `operand.field` addresses
the structured result. Comparisons use a scalar `value` or a declared `source_input`.
`source_text` is explanatory provenance, not a textual execution safety check.
Python parses the supported Markdown into `SourceProcedure` before the model
call. It owns metadata, input labels/identifiers/defaults/order, step headings,
normalized IDs/order/bodies and success text. The model returns only semantic
entries for the supplied IDs, with a source-sized output schema. Python checks
exact unique coverage and merges by ID in source order. Full LLM-authored
definitions, extra/missing semantic entries and unknown IDs are rejected.
The structured compiler interprets condition meaning once, without Python
classifying rule phrases, matching quotations, or inferring omissions from prose.
Stages distinguish `source_parsing`, `structured_output`, `pydantic` and
`deterministic_validation`. Source logs contain only identifiers/counts; semantic
logs omit comparison values, defaults, source prose and model reasoning.
The enrichment request explicitly separates `procedure_context` (title,
description and declared input descriptions) from each fixed step's ID, title and
complete source body. The prompt allows globally stated system context to apply
across steps and requires explicit control-flow rules to be preserved. System
values are never derived from resource-to-system mappings or the MCP catalog.
Semantic logs include argument names/references. For development, setting the
compiler logger to DEBUG additionally enables redacted `Semantic source` body
logs; these are absent at the normal INFO level.
Legacy `expression` fields are rejected, not interpreted. The supported operators
and runtime limits are listed in the README. Missing/future references, invalid
operator/value combinations and known incompatible bound result paths prevent
execution. Unavailable result paths at runtime produce `CONDITION_ERROR` and stop
the procedure without tool substitution.
