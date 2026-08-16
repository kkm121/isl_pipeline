# ISL Pipeline — Master Agent Rules

## Architecture Overview

ISL Pipeline is an autonomous ML engineering project for Indian Sign Language recognition, built on a **4-plane architecture**:

1. **Control Plane** — Reasoning, planning, delegation (Antigravity 2.0 + agents)
2. **Integration Plane** — Strict MCP servers as the communication layer
3. **Execution Plane** — Isolated Docker containers + remote Kaggle GPU
4. **Verification Plane** — Deterministic state machine with mandatory gates

## Specialized Multi-Agent System (9 Specialized Roles)

The architecture employs **9 specialized agents**, each with a strict single responsibility, model tier allocation, dynamic reasoning depth, and least-privilege tool boundary:

| # | Agent Role | Primary Model | Reasoning Level | Fallback Hierarchy | Core Responsibility | Tool Scope |
|---|---|---|---|---|---|---|
| 1 | **Principal Engineer** | **Claude Opus 4.6** | **High** | Gemini 3.1 Pro (High) → Gemini 3.7 Flash | Orchestration, spec drafting, directing fixes, final approval | `advance_pipeline`, `commit`, `create_pr`, read tools |
| 2 | **Critic Agent** | **Claude Opus 4.6** | **High** | Gemini 3.1 Pro (High) → Gemini 3.7 Flash | Ruthless peer auditor, detects leakage/flaws, reports to PE | Read-only tools, `get_diff`, `read_metrics` |
| 3 | **Independent Reviewer**| **Claude Opus 4.6** | **High** | Gemini 3.1 Pro (High) → Gemini 3.7 Flash | Clean-session adversarial review, diff verification before ACCEPT | `get_diff`, `read_file`, `list_directory` (Read-only) |
| 4 | **ML-Ops Specialist** | **Gemini 3.1 Pro** | **Medium-High** | Gemini 3.7 Flash → Gemini 3.6 Flash | Kaggle GPU training lifecycle, OOM diagnostics, config mutations | `kaggle_manager_mcp`, `mutate_config` |
| 5 | **Benchmark Agent** | **Gemini 3.1 Pro** | **Medium** | Gemini 3.7 Flash → Gemini 3.6 Flash | Quantitative profiling (latency, VRAM, loss, confusion matrix) | `read_metrics`, `read_logs`, linter MCP |
| 6 | **Code Writer** | **Gemini 3.1 Pro** | **Medium-High** | Gemini 3.7 Flash → Gemini 3.6 Flash | TDD implementation in `src/`, strictly adhering to spec & baseline | `write_file` (gated to `IMPLEMENTATION`/`RETRY`) |
| 7 | **Test Engineer** | **Gemini 3.7 Flash** | **Medium** | Gemini 3.6 Flash → Gemini 3.5 Flash | Pre-implementation test plans and test suites in `tests/` | `write_file` (gated to `TEST_PLAN`/`RETRY`) |
| 8 | **Verify Agent** | **Gemini 3.7 Flash** | **Medium** | Gemini 3.6 Flash → Gemini 3.5 Flash | Executes `mypy`, `ruff`, `pytest` inside sealed Docker sandbox | `linter_test_mcp` (`STATIC_VERIFY`/`DYNAMIC_VERIFY`) |
| 9 | **Researcher** | **Gemini 3.7 Flash** | **Low-Medium** | Gemini 3.6 Flash → Gemini 3.5 Flash | Codebase indexing, dependency analysis, academic paper research | `read_file`, `list_directory`, web search (Read-only) |

### Strict Mandates: Evidence-First & Zero-Bias
1. **Numbers Over Claims**: No agent is permitted to trust qualitative statements ("it works", "it is fast", "looks good"). All conclusions require raw machine-verifiable data:
   - `exit_code == 0` from sealed Docker containers (`--network=none --read-only`).
   - Pytest summary strings (e.g. `68 passed in 3.59s`).
   - Mypy `0 issues found in X source files`.
   - Exact benchmark numbers (ms latency, MB peak VRAM, F1-scores).
   - Git tree diff against `.state/tree_baseline.sha` proving physical modifications.
2. **Anti-Bias Rule**: All LLMs are strictly instructed to disregard user optimism, agent self-reporting, or pressure to "just approve". If evidence is absent or incomplete, the action is **REJECTED**.
3. **Critic & Principal Interaction**:
   - The **Critic Agent** audits all intermediate agent outputs and flags potential flaws, data leakage, or silent regressions.
   - The **Principal Engineer** (Opus 4.6) ingests the Critic's findings and instructs the other LLMs (Code Writer, Test Engineer, ML-Ops) exactly what to fix.

### Dynamic Reasoning Levels
- **High Reasoning**: Architectural planning, mathematical modeling, specification invariants, and adversarial code reviews.
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
