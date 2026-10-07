# Validate namespace application state

This procedure performs a series of checks against an OpenShift namespace.

## Procedure

**ID:** validate-namespace-application-state
**Version:** 1
**Risk:** low
**Confirmation required:** no

## Required information

- **Namespace** — required — OpenShift namespace to inspect.
- **Application name** — required — Name of the application to verify.
- **Minimum pod count** — optional — Defaults to `1`.

## Steps

### 1. Check that the namespace exists

Verify that **Namespace** exists in the OpenShift cluster.

If the namespace does not exist, stop the procedure and inform the user.

### 2. List pods in the namespace

Retrieve the pods running in **Namespace**.

### 3. Verify that enough pods exist

Check the pods returned by the previous step against **Minimum pod count**.

Continue only if the number of pods is greater than or equal to **Minimum pod count**.

If there are fewer pods than expected, stop the procedure and inform the user.

### 4. Get application details

Retrieve **Application name** from **Namespace**.

Run this step only if the pod-count validation succeeded.

### 5. Check recent namespace events

Retrieve recent warning and error events from **Namespace**.

### 6. Verify the application state

Verify that **Application name** exists in **Namespace** and that the previous checks completed successfully.

## Success

The procedure is successful when:

- the namespace exists,
- the number of pods is greater than or equal to **Minimum pod count**,
- the requested application exists,
- and the final application verification completes successfully.
