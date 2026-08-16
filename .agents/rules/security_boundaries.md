# Security Boundaries

This rule enforces the 4 strict security boundaries for the ISL Pipeline project.

### 1. Model Boundary
Agents have access to tools, resources, and outputs, but they MUST NEVER see raw credentials (such as API keys, authentication tokens, or passwords).

### 2. Filesystem Boundary
Agents are strictly restricted to accessing ONLY the project directory tree. They must not access or read the broader Windows filesystem or user directories outside the workspace.

### 3. Network Boundary
- Generated test code and local executions must run with NO network access (air-gapped).
- Integration tools and CI pipelines use strictly allowlisted endpoints only (e.g., `kaggle.com`, `github.com`, `pypi.org`).

### 4. Credential Boundary
All secrets must flow exclusively through the authorized credential broker. Secrets must never appear in agent context windows, git history, application logs, or generated artifacts.

### Violation Consequence
Violation of ANY of these security boundaries constitutes a CRITICAL failure and requires immediate human intervention.
