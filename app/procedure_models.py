"""Closed, non-executable representations of KB procedures."""
import re
import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

Text = Annotated[str, Field(min_length=1)]
Action = Literal["get", "list", "create", "update", "delete", "verify", "search"]


def implementation_specific(text: str) -> bool:
    """Bounded guard against obvious tool IDs/transport, not semantic verification."""
    return bool(re.search(
        r"\bmcp\b|mcp_|__\w+|\b(?:tool|capability)\s*:|\b[a-z][a-z0-9+.-]*://"
        r"|\b[a-z]\w*\.[a-z]\w*\.[a-z]\w*\b", text, re.I,
    ))


def identifier(text: str, separator: str = "_") -> str:
    return re.sub(r"[^a-z0-9]+", separator, text.lower()).strip(separator)


class ProcedureModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ProcedureInput(ProcedureModel):
    label: Text
    name: str = ""
    description: str | None = None
    required: bool = True
    default: str | int | bool | None = None

    @model_validator(mode="after")
    def normalize(self):
        normalized = identifier(self.label)
        if not normalized:
            raise ValueError("Input label has no supported identifier.")
        if self.name and self.name != normalized:
            raise ValueError("Input name must be the deterministic normalization of its label.")
        self.name = normalized
        return self


class StepArgument(ProcedureModel):
    name: Text
    source_input: Text


class ConditionOperand(ProcedureModel):
    step_id: Text
    # Dot-separated object keys only, relative to structured tool results.
    field: str | None = None

    @field_validator('field')
    @classmethod
    def closed_path(cls, value):
        if value is not None and not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z_][A-Za-z0-9_-]*)*', value):
            raise ValueError('Condition fields must be dot-separated object keys.')
        if value is not None and re.search(r'mcp_|__', value, re.I):
            raise ValueError('Implementation-specific condition fields are unsupported.')
        return value


class ProcedureCondition(ProcedureModel):
    operand: ConditionOperand
    operator: Literal['exists', 'not_exists', 'equals', 'not_equals', 'greater_than',
        'greater_than_or_equal', 'less_than', 'less_than_or_equal', 'is_true', 'is_false', 'is_empty', 'is_not_empty']
    value: str | int | float | bool | None = Field(default=None, description=
        'Comparison operators only: literal RHS, mutually exclusive with source_input. Must be null for all unary operators.')
    source_input: str | None = Field(default=None, description=
        'Comparison operators only: declared input name used as RHS, mutually exclusive with value. Must be null for all unary operators.')
    # Source provenance only; never interpreted by the runtime.
    source_text: Text

    @model_validator(mode='after')
    def valid_comparison(self):
        unary = self.operator in {'exists', 'not_exists', 'is_true', 'is_false', 'is_empty', 'is_not_empty'}
        if unary:
            if self.value is not None or self.source_input is not None:
                raise ValueError('Unary condition operators cannot have a value or input operand.')
        else:
            if (self.value is not None) == (self.source_input is not None):
                raise ValueError('Comparison operators require exactly one comparison value or declared input reference.')
            if self.source_input is not None and not self.source_input.strip():
                raise ValueError('Comparison input reference must be non-empty.')
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError('Condition values must be finite.')
        if self.operator in {'greater_than', 'greater_than_or_equal', 'less_than', 'less_than_or_equal'} and self.source_input is None and type(self.value) not in (int, float):
            raise ValueError('Ordered comparisons require a numeric value or input reference.')
        if not self.source_text.strip():
            raise ValueError('Condition source text must be non-empty.')
        return self


class ProcedureStep(ProcedureModel):
    id: str | None = None
    title: Text
    description: Text
    system: Text | None = None
    action: Action
    resource: Text
    arguments: list[StepArgument] = Field(default_factory=list)
    condition: ProcedureCondition | None = None
    stop_condition: ProcedureCondition | None = None

    @field_validator("id", "title", "description", "system", "action", "resource")
    @classmethod
    def semantic_only(cls, value):
        if value is not None and implementation_specific(value):
            raise PydanticCustomError("implementation_specific", "Implementation-specific identifiers or transport are unsupported.")
        return value

    @field_validator("system", "resource")
    @classmethod
    def normalized_resource(cls, value):
        if value is not None and not re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", value):
            raise PydanticCustomError("invalid_semantic_identifier", "Use a non-empty lowercase semantic identifier.")
        return value

    @field_validator("description")
    @classmethod
    def meaningful_description(cls, value):
        if not value.strip():
            raise ValueError("Step description must be non-empty.")
        return value

    @model_validator(mode="after")
    def normalize(self):
        if self.id is None:
            self.id = identifier(self.title)
        if not self.id or not self.id.strip():
            raise PydanticCustomError("invalid_step_id", "Step ID must be non-empty.", {"field": "id"})
        for argument in self.arguments:
            if implementation_specific(argument.name) or implementation_specific(argument.source_input):
                raise PydanticCustomError("implementation_specific", "Implementation-specific argument identifiers are unsupported.", {"field": "arguments", "step": self.id})
        for condition in (self.condition, self.stop_condition):
            if condition and (implementation_specific(condition.operand.step_id) or implementation_specific(condition.source_text)):
                raise PydanticCustomError("implementation_specific", "Implementation-specific condition fields are unsupported.", {"field": "conditions", "step": self.id})
        return self


class ProcedureDefinition(ProcedureModel):
    id: Text
    version: Annotated[int, Field(gt=0)]
    title: Text
    description: str | None = None
    risk: Literal["low", "medium", "high"]
    confirmation_required: bool
    inputs: list[ProcedureInput]
    steps: Annotated[list[ProcedureStep], Field(min_length=1)]
    success_criteria: str | None = None

    @model_validator(mode="after")
    def validate_references(self):
        if not self.id.strip() or not self.title.strip():
            raise ValueError("Procedure ID and title must be non-empty.")
        labels = [item.label for item in self.inputs]
        names = [item.name for item in self.inputs]
        ids = [step.id for step in self.steps]
        for index, item in enumerate(self.inputs):
            if item.label in labels[:index] or item.name in names[:index]:
                raise PydanticCustomError("duplicate_input", "Duplicate procedure inputs or normalized identifiers.", {"field": f"inputs.{index}.name", "reference": item.name})
        for index, step in enumerate(self.steps):
            if step.id in ids[:index]:
                raise PydanticCustomError("duplicate_step", "Duplicate step IDs.", {"field": f"steps.{index}.id", "step": step.id})
        previous = set()
        for index, step in enumerate(self.steps):
            arguments = [arg.name for arg in step.arguments]
            if len(set(arguments)) != len(arguments):
                raise PydanticCustomError("duplicate_argument", "Duplicate step arguments.", {"field": f"steps.{index}.arguments", "step": step.id})
            for arg_index, arg in enumerate(step.arguments):
                if arg.source_input not in names:
                    raise PydanticCustomError("unknown_input", "Unknown input reference.", {"field": f"steps.{index}.arguments.{arg_index}.source_input", "step": step.id, "reference": arg.source_input})
            for field, condition in (("condition", step.condition), ("stop_condition", step.stop_condition)):
                allowed = previous | {step.id} if field == "stop_condition" else previous
                if condition and condition.operand.step_id not in allowed:
                    if condition.operand.step_id not in ids:
                        reason = "Unknown step reference."
                    elif condition.operand.step_id == step.id:
                        reason = "Current step reference; preconditions must reference a previous step."
                    else:
                        reason = "Future step reference; conditions must not reference future steps."
                    raise PydanticCustomError("invalid_condition_reference", reason, {"field": f"steps.{index}.{field}.operand.step_id", "step": step.id, "reference": condition.operand.step_id})
                if condition and condition.source_input is not None and condition.source_input not in names:
                    raise PydanticCustomError('unknown_input', 'Unknown condition input reference.', {'field': f'steps.{index}.{field}.source_input', 'step': step.id, 'reference': condition.source_input})
                if condition and condition.source_input is not None:
                    item = next(item for item in self.inputs if item.name == condition.source_input)
                    if not item.required and item.default is None:
                        raise PydanticCustomError('invalid_condition', 'Comparison inputs must be required or have a declared default.', {'field': f'steps.{index}.{field}.source_input', 'step': step.id})
                    if item.default is not None and condition.operator in {'greater_than', 'greater_than_or_equal', 'less_than', 'less_than_or_equal'} and type(item.default) is not int:
                        raise PydanticCustomError('invalid_condition', 'Numeric comparisons require a numeric input default.', {'field': f'steps.{index}.{field}.source_input', 'step': step.id})
            previous.add(step.id)
        return self
