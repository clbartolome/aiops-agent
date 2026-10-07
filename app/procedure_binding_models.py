"""Environment-specific metadata and bindings; semantic models stay unchanged."""
from typing import Any, Literal
from hashlib import sha256
import json
import re

from pydantic import ConfigDict, Field, model_validator

from app.procedure_models import ProcedureDefinition, ProcedureModel, Text


class AvailableTool(ProcedureModel):
    server: Text
    name: Text
    description: str | None = None
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None
    annotations: dict[str, Any] = Field(default_factory=dict)
    operation_metadata: dict[str, Any] = Field(default_factory=dict)
    raw_name: str | None = None


ExecutionRisk = Literal['READ', 'WRITE', 'DESTRUCTIVE', 'UNKNOWN']


def execution_risk(tool: AvailableTool) -> ExecutionRisk:
    """Only explicit metadata establishes READ; unfamiliar tools stay UNKNOWN."""
    hints = tool.annotations
    signals = set()
    if hints.get('readOnlyHint') is True:
        signals.add('READ')
        words = set(re.split(r'[^a-z0-9]+', (tool.raw_name or tool.name).lower()))
        if words & {'create', 'add', 'update', 'modify', 'patch', 'delete', 'remove', 'provision'}:
            signals.add('WRITE')  # Contradictory metadata cannot establish READ.
    if hints.get('destructiveHint') is True:
        signals.add('DESTRUCTIVE')
    elif hints.get('readOnlyHint') is False:
        signals.add('WRITE')
    method = tool.operation_metadata.get('http_method')
    if isinstance(method, str):
        risk = {'GET': 'READ', 'HEAD': 'READ', 'OPTIONS': 'READ',
                'POST': 'WRITE', 'PUT': 'WRITE', 'PATCH': 'WRITE', 'DELETE': 'DESTRUCTIVE'}.get(method.upper())
        if risk:
            signals.add(risk)
    if not signals or ('READ' in signals and len(signals) > 1):
        return 'UNKNOWN'
    return 'DESTRUCTIVE' if 'DESTRUCTIVE' in signals else next(iter(signals))


def catalog_fingerprint(steps) -> str:
    # Fingerprint only relevant bound identities/schemas/risk, never auth or endpoints.
    entries = {}
    for step in steps:
        key = (step.mcp_server, step.tool_name)
        entry = dict(server=step.mcp_server, name=step.tool_name, input_schema=step.input_schema,
                     output_schema=step.output_schema, risk=step.execution_risk)
        if key in entries and entries[key] != entry:
            raise ValueError('A bound tool cannot have conflicting metadata or risk classifications.')
        entries[key] = entry
    def canonical(value):
        if isinstance(value, dict):
            return {key: sorted((canonical(item) for item in item_value), key=lambda item: json.dumps(item, sort_keys=True))
                    if key in {'required', 'enum'} and isinstance(item_value, list) else canonical(item_value)
                    for key, item_value in value.items()}
        if isinstance(value, list):
            return [canonical(item) for item in value]
        return value
    return sha256(json.dumps(canonical([entries[key] for key in sorted(entries)]), sort_keys=True,
                            separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class ToolArgumentBinding(ProcedureModel):
    procedure_argument: Text
    tool_argument: Text


class BoundProcedureStep(ProcedureModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)
    step_id: Text
    mcp_server: Text
    tool_name: Text
    argument_bindings: list[ToolArgumentBinding] = Field(default_factory=list)
    execution_risk: ExecutionRisk = 'UNKNOWN'
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] | None = None


class BoundProcedureDefinition(ProcedureModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)
    procedure: ProcedureDefinition
    steps: list[BoundProcedureStep]
    catalog_fingerprint: str = ''

    @model_validator(mode='after')
    def complete(self):
        if [step.step_id for step in self.steps] != [step.id for step in self.procedure.steps]:
            raise ValueError('Every semantic operation must have exactly one binding in source order.')
        for semantic, bound in zip(self.procedure.steps, self.steps):
            if semantic.system is None or bound.mcp_server != semantic.system:
                raise ValueError('Binding server must match the semantic system.')
            names = [binding.procedure_argument for binding in bound.argument_bindings]
            targets = [binding.tool_argument for binding in bound.argument_bindings]
            if len(names) != len(set(names)) or set(names) != {arg.name for arg in semantic.arguments}:
                raise ValueError('Every procedure argument must have exactly one mapping.')
            if len(targets) != len(set(targets)):
                raise ValueError('Tool arguments must have unique mappings.')
        fingerprint = catalog_fingerprint(self.steps)
        if self.catalog_fingerprint and self.catalog_fingerprint != fingerprint:
            raise ValueError('Bound catalog fingerprint does not match stored tool metadata.')
        object.__setattr__(self, 'catalog_fingerprint', fingerprint)
        return self


class ToolSelection(ProcedureModel):
    # Null explicitly means refusal/ambiguity; no runtime values are represented.
    tool_name: Text | None
    argument_bindings: list[ToolArgumentBinding]
    ambiguous: bool = False
