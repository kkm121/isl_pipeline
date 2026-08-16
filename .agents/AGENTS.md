# ISL Pipeline — Master Agent Rules

## Architecture Overview

ISL Pipeline is an autonomous ML engineering project for Indian Sign Language recognition, built on a **4-plane architecture**:

1. **Control Plane** — Reasoning, planning, delegation (Antigravity 2.0 + agents)
2. **Integration Plane** — Strict MCP servers as the communication layer
3. **Execution Plane** — Isolated Docker containers + remote Kaggle GPU
4. **Verification Plane** — Deterministic state machine with mandatory gates

## Model Allocation & Dynamic Reasoning Tiers

The architecture utilizes a tiered, high-capability model distribution with dynamic reasoning levels:

| Role | Primary Model | Reasoning Level | Fallback Hierarchy (on Limit Exhaustion) | Key Responsibilities |
|---|---|---|---|---|
| **Principal Engineer** | **Claude Opus 4.6** | **High** | Gemini 3.1 Pro (High) → Gemini 3.7 Flash | System architecture, specification gates, delegation |
| **Independent Reviewer** | **Claude Opus 4.6** | **High** | Gemini 3.1 Pro (High) → Gemini 3.7 Flash | Clean-session adversarial review, diff verification |
| **ML-Ops Specialist** | **Gemini 3.1 Pro** | **Medium-High** | Gemini 3.7 Flash → Gemini 3.6 Flash | Kaggle GPU lifecycle, OOM diagnostics, config mutation |
| **Test Engineer** | **Gemini 3.7 Flash** | **Medium** | Gemini 3.6 Flash → Gemini 3.5 Flash | TDD test suite generation, sealed Docker verification |
| **Researcher** | **Gemini 3.7 Flash** | **Low-Medium** | Gemini 3.6 Flash → Gemini 3.5 Flash | Codebase indexing, doc search, dependency analysis |

### Dynamic Reasoning Levels
- **High Reasoning**: Deep architectural planning, mathematical modeling, specification invariants, and adversarial code reviews.
- **Medium Reasoning**: GPU fault diagnostics, test plan edge-case generation, state transition validation.
- **Low Reasoning**: Code grep/search, fast log analysis, AST navigation, documentation retrieval.

### Graceful Fallback Policy
When any model reaches its rate limit or subscription window quota:
1. The orchestrator automatically cascades down to the next designated fallback tier.
2. If falling back to a lighter model (e.g. from Opus to 3.1 Pro or 3.7 Flash), the reasoning effort is dynamically adjusted up to preserve output quality.
3. State machine transitions and gate enforcement remain unchanged and strictly deterministic across all model tiers.

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
