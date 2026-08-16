# Hard Retry Limits — Mandatory Enforcement

## Default Limits
| Resource | Limit |
|---|---|
| max_retries_per_failure | 2 (same failure signature) |
| max_total_retries | 5 (across all failure types per task) |
| max_config_mutations | 3 (hyperparameter/config changes per task) |
| max_kaggle_submissions | 10 (per task) |
| max_concurrent_kaggle_jobs | 2 |

## Enforcement
- These limits are enforced by `src/orchestrator/state_machine.py` and `src/orchestrator/resource_budgets.py`
- When any limit is hit, the pipeline transitions to HUMAN_GATE automatically
- The LLM cannot override these limits

## Failure Signature Tracking
- Each failure is classified by its diagnostic signature (e.g., 'OOM', 'NaN', 'dependency_torch_version')
- If the same signature appears 2+ times: → HUMAN_GATE
- This prevents infinite loops where the agent tries the same fix repeatedly

## Escalation Path
```
failure detected
    ↓
diagnose & classify
    ↓
same signature seen before?
├── no → attempt remediation (retry count++)
└── yes (2x) → HUMAN_GATE
    ↓
retry count > max?
├── no → proceed
└── yes → HUMAN_GATE
```
