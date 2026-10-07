# Inspect namespace health

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

Retrieve recent events from **Namespace**, including warnings and errors when available.

### 4. Inspect deployments

Retrieve the deployments configured in **Namespace** and their current state.

## Success

The procedure is successful when the namespace exists and the pod, event, and deployment information has been retrieved successfully.
