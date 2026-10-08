# Inspect namespace health

Inspect the current state of an OpenShift namespace and report basic workload health.
Read-only: this procedure never modifies cluster state.

## Procedure

**ID:** inspect-namespace-health
**Version:** 1
**Risk:** low
**Confirmation required:** no

## Required information

- **Namespace** — required — OpenShift namespace to inspect.

## Steps

### 1. Verify that the namespace exists

Check that **Namespace** exists in the OpenShift cluster. If it does not exist, stop the
procedure and report that the namespace was not found. Do not proceed to later steps.

### 2. List pods in the namespace

Retrieve all pods running in **Namespace** and summarize their status (e.g. Running,
Pending, CrashLoopBackOff).

### 3. Review recent events

Retrieve recent events for **Namespace** and summarize anything that looks abnormal.

## Success

The namespace exists and its pod status and recent events have been retrieved successfully.
