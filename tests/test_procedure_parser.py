import pytest

from app.procedure.models import ProcedureDefinition
from app.procedure.parser import ProcedureParseError, normalize_identifier, parse_procedure_markdown

FULL_EXAMPLE = """# Inspect namespace health

Inspect the current state of an OpenShift namespace and report basic workload health.

## Procedure

**ID:** inspect-namespace-health
**Version:** 1
**Risk:** low
**Confirmation required:** no

## Required information

- **Namespace** — required — OpenShift namespace to inspect.

## Steps

### 1. Verify that the namespace exists

Check that **Namespace** exists in the OpenShift cluster.

If the namespace does not exist, stop the procedure and inform the user.

### 2. List pods in the namespace

Retrieve all pods running in **Namespace**.

### 3. Review recent events

Retrieve recent events from **Namespace**.

## Success

The namespace exists and the requested information has been retrieved successfully.
"""


def make_markdown(
    *, title="Inspect namespace health", procedure_id="inspect-namespace-health",
    version="1", risk="low", confirmation="no",
    inputs_section="- **Namespace** — required — OpenShift namespace to inspect.",
    steps_section=(
        "### 1. Verify that the namespace exists\n\n"
        "Check that **Namespace** exists in the OpenShift cluster.\n\n"
        "### 2. List pods in the namespace\n\n"
        "Retrieve all pods running in **Namespace**.\n"
    ),
    success_section="The namespace exists and the requested information has been retrieved successfully.",
):
    return f"""# {title}

Inspect the current state of an OpenShift namespace and report basic workload health.

## Procedure

**ID:** {procedure_id}
**Version:** {version}
**Risk:** {risk}
**Confirmation required:** {confirmation}

## Required information

{inputs_section}

## Steps

{steps_section}

## Success

{success_section}
"""


# --- Full example -----------------------------------------------------------

def test_full_example_parses_into_a_complete_procedure_definition():
    definition = parse_procedure_markdown(FULL_EXAMPLE)

    assert isinstance(definition, ProcedureDefinition)
    assert definition.title == "Inspect namespace health"
    assert definition.description == (
        "Inspect the current state of an OpenShift namespace and report basic workload health."
    )
    assert definition.id == "inspect-namespace-health"
    assert definition.version == 1
    assert definition.risk == "low"
    assert definition.confirmation_required is False
    assert [item.name for item in definition.inputs] == ["namespace"]
    assert [step.id for step in definition.steps] == [
        "verify_that_the_namespace_exists", "list_pods_in_the_namespace", "review_recent_events",
    ]
    assert all(step.input_refs == ["namespace"] for step in definition.steps)
    assert definition.success_criteria == (
        "The namespace exists and the requested information has been retrieved successfully."
    )


# --- normalize_identifier ----------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("Namespace", "namespace"),
    ("Application name", "application_name"),
    ("Minimum pod count", "minimum_pod_count"),
    ("Verify that the namespace exists", "verify_that_the_namespace_exists"),
    ("List pods in the namespace", "list_pods_in_the_namespace"),
])
def test_normalize_identifier_is_deterministic(text, expected):
    assert normalize_identifier(text) == expected
    assert normalize_identifier(text) == normalize_identifier(text)  # same input -> same output


# --- Metadata parsing ---------------------------------------------------------

def test_metadata_id_version_risk_and_confirmation_are_parsed():
    definition = parse_procedure_markdown(make_markdown(
        procedure_id="inspect-namespace-health", version="3", risk="high", confirmation="yes",
    ))
    assert definition.id == "inspect-namespace-health"
    assert definition.version == 3
    assert definition.risk == "high"
    assert definition.confirmation_required is True


@pytest.mark.parametrize("confirmation, expected", [
    ("no", False), ("No", False), ("false", False), ("False", False), ("0", False),
    ("yes", True), ("Yes", True), ("true", True), ("True", True), ("1", True),
])
def test_confirmation_required_boolean_words_are_supported(confirmation, expected):
    definition = parse_procedure_markdown(make_markdown(confirmation=confirmation))
    assert definition.confirmation_required is expected


@pytest.mark.parametrize("risk", ["low", "medium", "high"])
def test_supported_risk_levels_are_accepted(risk):
    assert parse_procedure_markdown(make_markdown(risk=risk)).risk == risk


def test_missing_metadata_field_fails_clearly():
    markdown = FULL_EXAMPLE.replace("**Risk:** low\n", "")
    with pytest.raises(ProcedureParseError, match="Risk"):
        parse_procedure_markdown(markdown)


def test_unsupported_risk_fails():
    with pytest.raises(ProcedureParseError, match="Risk"):
        parse_procedure_markdown(make_markdown(risk="critical"))


def test_non_integer_version_fails():
    with pytest.raises(ProcedureParseError, match="Version"):
        parse_procedure_markdown(make_markdown(version="one"))


def test_zero_version_fails():
    with pytest.raises(ProcedureParseError, match="Version"):
        parse_procedure_markdown(make_markdown(version="0"))


def test_empty_id_fails():
    with pytest.raises(ProcedureParseError, match="ID"):
        parse_procedure_markdown(make_markdown(procedure_id=""))


def test_unsupported_confirmation_value_fails():
    with pytest.raises(ProcedureParseError, match="Confirmation"):
        parse_procedure_markdown(make_markdown(confirmation="maybe"))


# --- Inputs -------------------------------------------------------------------

def test_required_input_is_parsed_with_label_name_and_required_flag():
    definition = parse_procedure_markdown(make_markdown(
        inputs_section="- **Namespace** — required — OpenShift namespace to inspect.",
        steps_section="### 1. Verify that the namespace exists\n\nCheck **Namespace**.\n",
    ))
    assert len(definition.inputs) == 1
    item = definition.inputs[0]
    assert item.label == "Namespace"
    assert item.name == "namespace"
    assert item.required is True
    assert item.default is None


@pytest.mark.parametrize("raw_default, expected", [
    ("1", 1),
    ("3.5", 3.5),
    ("true", True),
    ("false", False),
    ("operations", "operations"),
])
def test_optional_input_default_is_parsed_by_type(raw_default, expected):
    definition = parse_procedure_markdown(make_markdown(
        inputs_section=(
            f"- **Minimum pod count** — optional — Minimum expected number. Defaults to `{raw_default}`."
        ),
        steps_section="### 1. Check\n\nCheck **Minimum pod count**.\n",
    ))
    item = definition.inputs[0]
    assert item.name == "minimum_pod_count"
    assert item.required is False
    assert item.default == expected
    assert type(item.default) is type(expected)


def test_default_without_backtick_value_fails():
    with pytest.raises(ProcedureParseError, match="`value`"):
        parse_procedure_markdown(make_markdown(
            inputs_section="- **Minimum pod count** — optional — Minimum expected number. Defaults to 1.",
            steps_section="### 1. Check\n\nCheck **Minimum pod count**.\n",
        ))


def test_multiple_inputs_preserve_source_order():
    definition = parse_procedure_markdown(make_markdown(
        inputs_section=(
            "- **Namespace** — required — OpenShift namespace to inspect.\n"
            "- **Application name** — required — Application to inspect.\n"
            "- **Minimum pod count** — optional — Minimum expected number. Defaults to `1`."
        ),
        steps_section=(
            "### 1. Check\n\nCheck **Namespace**, **Application name** and **Minimum pod count**.\n"
        ),
    ))
    assert [item.name for item in definition.inputs] == [
        "namespace", "application_name", "minimum_pod_count",
    ]


def test_duplicate_input_names_fail():
    with pytest.raises(ProcedureParseError, match="Duplicate input"):
        parse_procedure_markdown(make_markdown(
            inputs_section=(
                "- **Namespace** — required — OpenShift namespace to inspect.\n"
                "- **namespace** — optional — A duplicate under a different case."
            ),
            steps_section="### 1. Check\n\nCheck **Namespace**.\n",
        ))


# --- Steps ----------------------------------------------------------------

def test_only_numbered_step_headings_under_steps_become_steps():
    definition = parse_procedure_markdown(FULL_EXAMPLE)
    assert len(definition.steps) == 3
    assert [step.title for step in definition.steps] == [
        "Verify that the namespace exists", "List pods in the namespace", "Review recent events",
    ]


def test_step_ordering_follows_markdown_order():
    definition = parse_procedure_markdown(make_markdown(
        steps_section=(
            "### 1. Zebra step\n\nCheck **Namespace**.\n\n"
            "### 2. Alpha step\n\nCheck **Namespace**.\n"
        ),
    ))
    assert [step.title for step in definition.steps] == ["Zebra step", "Alpha step"]


def test_step_ids_are_generated_deterministically_from_titles():
    definition = parse_procedure_markdown(FULL_EXAMPLE)
    assert [step.id for step in definition.steps] == [
        "verify_that_the_namespace_exists", "list_pods_in_the_namespace", "review_recent_events",
    ]


def test_step_body_preserves_complete_instruction_text():
    definition = parse_procedure_markdown(FULL_EXAMPLE)
    first = definition.steps[0]
    assert first.instruction == (
        "Check that **Namespace** exists in the OpenShift cluster.\n\n"
        "If the namespace does not exist, stop the procedure and inform the user."
    )


def test_steps_section_without_any_step_heading_fails():
    with pytest.raises(ProcedureParseError, match="Steps"):
        parse_procedure_markdown(make_markdown(steps_section="Just some prose, no step headings."))


def test_missing_steps_section_fails():
    markdown = FULL_EXAMPLE.replace("## Steps", "## Not Steps")
    with pytest.raises(ProcedureParseError, match="Steps"):
        parse_procedure_markdown(markdown)


def test_duplicate_step_ids_fail():
    with pytest.raises(ProcedureParseError, match="Duplicate step"):
        parse_procedure_markdown(make_markdown(
            steps_section=(
                "### 1. Check namespace\n\nCheck **Namespace**.\n\n"
                "### 2. Check namespace\n\nCheck **Namespace** again.\n"
            ),
        ))


def test_other_markdown_headings_are_not_treated_as_steps():
    definition = parse_procedure_markdown(FULL_EXAMPLE)
    titles = [step.title for step in definition.steps]
    assert "Procedure" not in titles
    assert "Required information" not in titles
    assert "Success" not in titles
    assert len(definition.steps) == 3


def test_numbered_prose_inside_a_step_body_does_not_create_extra_steps():
    definition = parse_procedure_markdown(make_markdown(
        steps_section=(
            "### 1. Do the checks\n\n"
            "Perform the following in order on **Namespace**:\n\n"
            "1. Check pods\n"
            "2. Check events\n"
            "3. Check deployments\n"
        ),
    ))
    assert len(definition.steps) == 1
    assert "1. Check pods" in definition.steps[0].instruction
    assert "3. Check deployments" in definition.steps[0].instruction


# --- Input references -------------------------------------------------------

def test_bold_references_to_declared_inputs_are_extracted():
    definition = parse_procedure_markdown(make_markdown(
        inputs_section=(
            "- **Namespace** — required — OpenShift namespace to inspect.\n"
            "- **Application name** — required — Application to inspect."
        ),
        steps_section=(
            "### 1. Retrieve application\n\nRetrieve **Application name** from **Namespace**.\n"
        ),
    ))
    assert definition.steps[0].input_refs == ["application_name", "namespace"]


def test_repeated_reference_to_the_same_input_is_not_duplicated():
    definition = parse_procedure_markdown(make_markdown(
        steps_section=(
            "### 1. Check twice\n\nCheck **Namespace** and then re-check **Namespace** again.\n"
        ),
    ))
    assert definition.steps[0].input_refs == ["namespace"]


def test_step_without_any_bold_reference_has_no_input_refs():
    definition = parse_procedure_markdown(make_markdown(
        steps_section="### 1. Say hello\n\nSay hello to the operator.\n",
    ))
    assert definition.steps[0].input_refs == []


def test_unknown_bold_input_reference_fails_parsing():
    with pytest.raises(ProcedureParseError, match="undeclared input"):
        parse_procedure_markdown(make_markdown(
            steps_section=(
                "### 1. Check something\n\nRetrieve **Workload name** from **Namespace**.\n"
            ),
        ))


# --- Success ------------------------------------------------------------------

def test_success_section_is_preserved_as_human_readable_text():
    definition = parse_procedure_markdown(make_markdown(
        success_section="Everything checked out fine.",
    ))
    assert definition.success_criteria == "Everything checked out fine."


def test_missing_success_section_is_none():
    markdown = FULL_EXAMPLE.split("## Success")[0]
    definition = parse_procedure_markdown(markdown)
    assert definition.success_criteria is None
