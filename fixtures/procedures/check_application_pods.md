# Check application pods

Check the pods belonging to one application in one namespace. Read-only: this procedure
never modifies cluster state.

## Procedure

**ID:** check-application-pods
**Version:** 1
**Risk:** low
**Confirmation required:** no

## Required information

- **Namespace** — required — OpenShift namespace that hosts the application.
- **Application name** — required — Name of the application to check.

## Steps

### 1. List application pods

List the pods for **Application name** in **Namespace** and summarize their status.

### 2. Check application logs

Retrieve recent logs for **Application name** in **Namespace** and summarize any errors.

## Success

Pod status and recent logs for **Application name** in **Namespace** have been retrieved
successfully.
