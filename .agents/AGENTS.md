# Master Rules: ISL Pipeline

## Project Overview
ISL Pipeline is an autonomous Machine Learning engineering project dedicated to Indian Sign Language recognition.

## Architecture Mandate
All work on this project MUST strictly follow the 4-plane architecture:
1. **Control Plane**
2. **Integration Plane**
3. **Execution Plane**
4. **Verification Plane**

## Principal Engineer
The Principal Engineer is Claude Opus 4.6. This agent is responsible for reasoning, planning, delegating tasks, evaluating evidence, and making final engineering decisions.

## Evidence-Based Verification
Agents must NEVER trust self-reported success claims. Verification requires concrete evidence:
- Actual pytest output
- Actual mypy output
- Actual git diff evidence

## Specification Gate
No implementation is permitted without passing the Specification Gate. This requires:
1. Precise specification
2. Machine-verifiable acceptance criteria
3. Tests defined *before* implementation

## Agent Roles
- **Principal Engineer**: Orchestration and decision making.
- **Researcher**: Read-only codebase and literature exploration.
- **Test Engineer**: Test creation, CI execution, static analysis.
- **Reviewer**: Adversarial code review.
- **ML-Ops**: Kaggle pipeline management and remote diagnostics.

## Security
- **No Credentials**: No secrets or credentials may appear in context, code, or logs.
- **Filesystem Restriction**: No unrestricted filesystem access; operate only within the project directory.
- **Sandboxing**: All generated code must be executed in a sealed sandbox.

## MCP-First Interaction
Agents must prefer utilizing specific MCP tools and servers over executing arbitrary shell commands.

## Verification Pipeline
All code changes must pass the rigorous 7-step verification process before being considered for merge.
