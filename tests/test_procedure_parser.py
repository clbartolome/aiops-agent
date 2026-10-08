import pytest

from app.procedure.parser import ProcedureParseError, normalize_name, parse_procedure

VALID_KB = """# Inspect namespace health

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

## Success

The procedure is successful when the namespace exists and pods have been retrieved.
"""

MULTI_INPUT_KB = """# Validate application state

Validate that an application exists in an OpenShift namespace and inspect its runtime state.

## Procedure

**ID:** validate-application-state
**Version:** 2
**Risk:** medium
**Confirmation required:** yes

## Required information

- **Namespace** — required — OpenShift namespace containing the application.
- **Application name** — required — Name of the application to inspect.
- **Minimum pod count** — optional — Minimum expected number of running pods. Defaults to `1`.

## Steps

### 1. Verify that the namespace exists

Check that **Namespace** exists.

### 2. Retrieve the application

Retrieve **Application name** from **Namespace**.

1. First check the deployment.
2. Then check the replica set.

If the application does not exist, stop the procedure and inform the user.

## Success

The procedure is successful when the application and its pods have been validated.
"""


# --- Title / description -------------------------------------------------

def test_parses_title_preserving_source_wording():
    procedure = parse_procedure(VALID_KB)
    assert procedure.title == "Inspect namespace health"


def test_parses_description_between_title_and_procedure_section():
    procedure = parse_procedure(VALID_KB)
    assert procedure.description == (
        "Inspect the current state of an OpenShift namespace and report basic workload health."
    )


def test_missing_title_fails():
    content = "## Procedure\n\n**ID:** x\n**Version:** 1\n**Risk:** low\n**Confirmation required:** no\n\n## Steps\n\n### 1. Do it\n\nBody.\n"
    with pytest.raises(ProcedureParseError):
        parse_procedure(content)


# --- Metadata --------------------------------------------------------------

def test_parses_valid_metadata():
    procedure = parse_procedure(VALID_KB)
    assert procedure.id == "inspect-namespace-health"
    assert procedure.version == 1
    assert procedure.risk == "low"
    assert procedure.confirmation_required is False


def test_risk_is_case_normalized():
    content = VALID_KB.replace("**Risk:** low", "**Risk:** LOW")
    procedure = parse_procedure(content)
    assert procedure.risk == "low"


def test_confirmation_required_accepts_yes_no():
    content = VALID_KB.replace("**Confirmation required:** no", "**Confirmation required:** yes")
    procedure = parse_procedure(content)
    assert procedure.confirmation_required is True


def test_confirmation_required_accepts_true_false():
    content = VALID_KB.replace("**Confirmation required:** no", "**Confirmation required:** true")
    procedure = parse_procedure(content)
    assert procedure.confirmation_required is True


def test_missing_id_fails():
    content = VALID_KB.replace("**ID:** inspect-namespace-health\n", "")
    with pytest.raises(ProcedureParseError, match="ID"):
        parse_procedure(content)


def test_invalid_version_fails():
    content = VALID_KB.replace("**Version:** 1", "**Version:** zero")
    with pytest.raises(ProcedureParseError, match="version"):
        parse_procedure(content)


def test_non_positive_version_fails():
    content = VALID_KB.replace("**Version:** 1", "**Version:** 0")
    with pytest.raises(ProcedureParseError, match="version"):
        parse_procedure(content)


def test_missing_version_fails():
    content = VALID_KB.replace("**Version:** 1\n", "")
    with pytest.raises(ProcedureParseError, match="Version"):
        parse_procedure(content)


def test_invalid_risk_fails():
    content = VALID_KB.replace("**Risk:** low", "**Risk:** extreme")
    with pytest.raises(ProcedureParseError, match="risk"):
        parse_procedure(content)


def test_invalid_confirmation_required_fails():
    content = VALID_KB.replace("**Confirmation required:** no", "**Confirmation required:** maybe")
    with pytest.raises(ProcedureParseError, match="confirmation required"):
        parse_procedure(content)


# --- Required information / inputs ----------------------------------------

def test_parses_required_input():
    procedure = parse_procedure(VALID_KB)
    assert len(procedure.inputs) == 1
    item = procedure.inputs[0]
    assert item.name == "namespace"
    assert item.label == "Namespace"
    assert item.required is True
    assert item.default is None
    assert item.description == "OpenShift namespace to inspect."


def test_parses_optional_input_with_default():
    procedure = parse_procedure(MULTI_INPUT_KB)
    optional = next(item for item in procedure.inputs if item.name == "minimum_pod_count")
    assert optional.required is False
    assert optional.default == 1
    assert optional.label == "Minimum pod count"


def test_multiple_inputs_preserve_source_order():
    procedure = parse_procedure(MULTI_INPUT_KB)
    assert [item.name for item in procedure.inputs] == [
        "namespace", "application_name", "minimum_pod_count",
    ]


def test_duplicate_input_names_fail():
    content = MULTI_INPUT_KB.replace(
        "- **Application name** — required — Name of the application to inspect.\n",
        "- **Namespace** — required — Duplicate label normalizes the same.\n",
    )
    with pytest.raises(ProcedureParseError, match="Duplicate input name"):
        parse_procedure(content)


@pytest.mark.parametrize("default_text, expected", [
    ("Defaults to `1`.", 1),
    ("Defaults to `1.5`.", 1.5),
    ("Defaults to `true`.", True),
    ("Defaults to `prod`.", "prod"),
])
def test_default_parsing_supports_primitive_types(default_text, expected):
    content = VALID_KB.replace(
        "- **Namespace** — required — OpenShift namespace to inspect.\n",
        f"- **Namespace** — required — OpenShift namespace to inspect. {default_text}\n",
    )
    procedure = parse_procedure(content)
    assert procedure.inputs[0].default == expected


def test_no_required_information_section_means_no_inputs():
    content = VALID_KB.replace(
        "## Required information\n\n- **Namespace** — required — OpenShift namespace to inspect.\n\n",
        "",
    )
    procedure = parse_procedure(content)
    assert procedure.inputs == []


# --- Steps -------------------------------------------------------------------

def test_only_numbered_step_headings_become_steps():
    procedure = parse_procedure(VALID_KB)
    assert len(procedure.steps) == 2
    assert [step.title for step in procedure.steps] == [
        "Verify that the namespace exists", "List pods in the namespace",
    ]


def test_steps_preserve_source_order():
    procedure = parse_procedure(MULTI_INPUT_KB)
    assert [step.title for step in procedure.steps] == [
        "Verify that the namespace exists", "Retrieve the application",
    ]


def test_step_instruction_is_preserved_verbatim_including_conditionals():
    procedure = parse_procedure(VALID_KB)
    first = procedure.steps[0]
    assert first.instruction == (
        "Check that **Namespace** exists in the OpenShift cluster.\n\n"
        "If the namespace does not exist, stop the procedure and inform the user."
    )


def test_numbered_prose_inside_instruction_does_not_create_new_steps():
    procedure = parse_procedure(MULTI_INPUT_KB)
    assert len(procedure.steps) == 2
    second = procedure.steps[1]
    assert "1. First check the deployment." in second.instruction
    assert "2. Then check the replica set." in second.instruction


def test_step_ids_are_generated_from_titles():
    procedure = parse_procedure(VALID_KB)
    assert procedure.steps[0].id == "verify_that_the_namespace_exists"
    assert procedure.steps[1].id == "list_pods_in_the_namespace"


def test_duplicate_step_titles_produce_duplicate_ids_and_fail():
    content = VALID_KB.replace(
        "### 2. List pods in the namespace",
        "### 2. Verify that the namespace exists",
    )
    with pytest.raises(ProcedureParseError, match="Duplicate step id"):
        parse_procedure(content)


def test_no_steps_fails():
    content = VALID_KB.replace(
        "### 1. Verify that the namespace exists\n\n"
        "Check that **Namespace** exists in the OpenShift cluster.\n\n"
        "If the namespace does not exist, stop the procedure and inform the user.\n\n"
        "### 2. List pods in the namespace\n\n"
        "Retrieve all pods running in **Namespace**.\n\n",
        "",
    )
    with pytest.raises(ProcedureParseError, match="steps"):
        parse_procedure(content)


# --- Success criteria --------------------------------------------------------

def test_success_criteria_preserved_verbatim():
    procedure = parse_procedure(VALID_KB)
    assert procedure.success_criteria == (
        "The procedure is successful when the namespace exists and pods have been retrieved."
    )


def test_success_criteria_absent_when_no_success_section():
    content = VALID_KB.split("## Success")[0]
    procedure = parse_procedure(content)
    assert procedure.success_criteria is None


# --- normalize_name ----------------------------------------------------------

@pytest.mark.parametrize("label, expected", [
    ("Minimum pod count", "minimum_pod_count"),
    ("Namespace", "namespace"),
    ("  Trim Me  ", "trim_me"),
    ("Weird--Spacing!!", "weird_spacing"),
])
def test_normalize_name(label, expected):
    assert normalize_name(label) == expected
