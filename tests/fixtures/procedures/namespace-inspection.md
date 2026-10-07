# Inspect namespace

Inspect application information in OpenShift.

## Procedure

**ID:** inspect-namespace
**Version:** 1
**Risk:** low
**Confirmation required:** no

## Required information

- **Namespace** — required — Namespace to inspect.
- **Application name** — required — Application to retrieve.

## Steps

### 1. Get namespace

**ID:** namespace-check

Get the namespace identified by **Namespace** in OpenShift.

If the namespace check reports that it failed, stop the procedure.

### 2. Get application

**ID:** application-check

Get **Application name** from **Namespace**.

Continue only if the namespace check reports that it did not fail.

### 3. List events

**ID:** events-list

List recent events from **Namespace**.

## Success

The requested namespace information has been retrieved.
