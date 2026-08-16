# Evidence-Based Verification

This rule enforces the evidence-based verification standard for the project.

The Principal Engineer (and all agents) MUST NOT trust another agent's statement of "success" or "it works". All claims must be backed by concrete evidence.

### Required Evidence Types

Depending on the task, the following evidence must be provided:

- **Test Results**: Actual raw `pytest` output showing pass/fail counts and log details.
- **Static Analysis**: Actual `mypy` or `ruff` terminal output.
- **Git Diff**: The actual `git diff` output demonstrating precisely what changed.
- **Metrics**: Actual metric values printed from evaluation runs.
- **Artifacts**: Concrete file paths, sizes, and hashes of produced outputs (e.g., model weights, logs).

### Rejection Protocol
If an agent reports "all tests pass" or "code is fixed" without providing the actual `pytest` or command output in their message, the claim is automatically REJECTED.

### HEAD Verification
Always verify that `git HEAD` matches the expected state after claimed changes are applied.
