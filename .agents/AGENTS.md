# ISL Pipeline — Master Agent Rules

## Architecture Overview

ISL Pipeline is an autonomous ML engineering project for Indian Sign Language recognition, built on a **4-plane architecture**:

1. **Control Plane** — Reasoning, planning, delegation (Antigravity 2.0 + agents)
2. **Integration Plane** — Strict MCP servers as the communication layer
3. **Execution Plane** — Isolated Docker containers + remote Kaggle GPU
4. **Verification Plane** — Deterministic state machine with mandatory gates

## Model-Agnostic Design (Correction #1)

The architecture is **model-agnostic** at the infrastructure layer. The Principal Engineer role is filled by whichever model the user configures.

- **Current selection**: Claude Opus 4.6
- **The infrastructure does NOT assume** any specific model's capabilities
- **All model interaction** flows through the MCP tool layer

## Agent Roles with Least-Privilege Permissions (Correction #15)

| Agent | Model | Permissions | Tools |
|---|---|---|---|
| Principal Engineer | Configurable (current: Opus 4.6) | Full orchestration | All MCP servers, state machine |
| Researcher | Fast model (read-only) | Read-only codebase access | Filesystem MCP (read), web search |
| Test Engineer | Fast model | Test execution only | Linter/Test MCP, Filesystem MCP |
| Reviewer | Configurable (current: Opus 4.6) | Read-only + diff | Filesystem MCP (read), GitHub MCP (diff) |
| ML-Ops | Pro model | Kaggle execution only | Kaggle Manager MCP |

**Least privilege**: Each agent gets ONLY the tools it needs. Researcher and Reviewer are read-only.

## Deterministic State Machine (Correction #4)

The pipeline workflow is enforced by a **hardcoded state machine**, not by the LLM:

```
IDLE → SPECIFICATION → TEST_PLAN → IMPLEMENTATION → STATIC_VERIFY 
  → DYNAMIC_VERIFY → DIFF_VERIFY → INDEPENDENT_REVIEW → ACCEPT → COMPLETE
```

- The LLM provides evidence and decisions
- The **transition logic is hardcoded** in `src/orchestrator/state_machine.py`
- The LLM **cannot skip steps**, bypass verification, or self-approve
- Failed verifications → RETRY (with limits) or HUMAN_GATE

## HUMAN_GATE (Correction #5)

The pipeline has an explicit terminal state for situations requiring human intervention:
- Uncertain diagnosis (confidence < 0.7)
- Same failure repeated 2+ times
- Destructive operations
- Credential issues
- Policy/ToS concerns
- Resource budget exceeded
- Max retries exceeded

See `.agents/rules/human_gate.md` for full specification.

## Evidence-Based Verification

- NEVER trust self-reported success
- Require actual pytest output, mypy output, git diff evidence
- HEAD-diff check: if agent claims changes but diff shows zero lines changed → REJECT

## Specification Gate

- No implementation without precise spec + acceptance criteria + tests
- Acceptance criteria must be machine-verifiable
- See `.agents/rules/specification_gate.md`

## Security Boundaries (Correction #3)

### Credential Broker — NOT .env files

Credentials flow through the **Account Broker** (`kaggle/account_broker.py`):
```
Credential Broker
    ├── Kaggle account 1 secret
    ├── Kaggle account 2 secret  
    ├── Kaggle account 3 secret
    ├── GitHub credential
    └── Hugging Face credential
```

The MCP server asks the broker to **perform an authenticated operation**.
Raw tokens **NEVER** enter:
- Agent context/prompts
- Docker containers (injected at runtime via KAGGLE_CONFIG_DIR)
- Git repository
- Logs
- Generated artifacts

### Four Boundaries
1. **Model boundary**: Agents see tools/resources/results, never raw credentials
2. **Filesystem boundary**: Agents access ONLY the project directory
3. **Network boundary**: Sealed sandbox (no network) / Integration runner (controlled network) / Kaggle (remote)
4. **Credential boundary**: All secrets through broker, never in context

## MCP-First Interaction

Prefer MCP tools over arbitrary shell commands. The 4 MCP servers are:
1. **Filesystem MCP** — read/write files within project boundary
2. **GitHub MCP** — issues, branches, commits, PRs, diffs
3. **Linter/Test MCP** — pytest, mypy, ruff with structured output
4. **Kaggle Manager MCP** — kernel lifecycle with account broker

## Resource Budgets (Correction #10)

All operations are subject to hard resource limits enforced by `src/orchestrator/resource_budgets.py`.
See `.agents/rules/resource_budgets.md`.

## Retry Limits (Correction #6)

Hard retry limits prevent infinite loops. Same failure 2x → HUMAN_GATE.
See `.agents/rules/retry_limits.md`.

## Kaggle Policy Gate (Correction #7)

Account rotation is NOT unconditional:
```
quota exhausted?
    ↓
is account rotation permitted?
├── yes → next account
└── no → HUMAN_GATE
```

## Antigravity-Native Integration (Correction #14)

This architecture runs ON TOP of Antigravity 2.0, using its native capabilities:
- **Projects**: Workspace-scoped agent configuration
- **Subagents**: define_subagent / invoke_subagent for role-based delegation  
- **Worktrees**: Isolated git worktrees for parallel agent work
- **MCP**: Native MCP server registration via .agents/mcp_config.json
- **Hooks**: Lifecycle hooks via .agents/hooks.json
- **Scheduling**: Built-in async/scheduled task execution
- **Permissions**: Inherited permission model with per-agent restrictions

We do NOT recreate a competing orchestrator underneath Antigravity.
