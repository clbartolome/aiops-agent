# Launch AAP job

Launch an existing Ansible Automation Platform job template against a target host and
confirm it completed successfully. This procedure performs a write operation and requires
confirmation before it starts.

## Procedure

**ID:** launch-aap-job
**Version:** 1
**Risk:** medium
**Confirmation required:** yes

## Required information

- **Job template name** — required — Name of the existing AAP job template to launch.

## Steps

### 1. Find job template

Find the job template named **Job template name** in AAP. If it does not exist, stop the
procedure and report that the job template was not found.

### 2. Review job template

Review the job template's configuration (inventory, playbook, extra variables) and
summarize it before launching.

### 3. Launch job template

Launch the job template **Job template name**. This is a write operation and requires
approval before the launch is executed.

### 4. Verify job

Check the status of the launched job and report whether it completed successfully.

## Success

The job template was launched and completed successfully.
