# Resource Budget Enforcement

All autonomous operations are subject to resource budgets tracked by `src/orchestrator/resource_budgets.py`.

## Budget Defaults
| Resource | Limit | Rationale |
|---|---|---|
| Runtime per task | 1 hour | Prevent runaway tasks |
| Total retries | 5 | Prevent infinite loops |
| Kaggle submissions | 10 | Conserve GPU quota |
| Disk usage | 10 GB | Prevent disk fill |
| Artifact size | 1 GB per artifact | Prevent oversized outputs |
| Container memory | 4 GB | Prevent OOM on host |
| Container CPU | 2 cores | Fair host resource sharing |
| Concurrent agents | 5 | Prevent context/resource explosion |
| Concurrent Kaggle jobs | 2 | Respect platform limits |

## Enforcement
- ResourceTracker checks budgets before every resource-consuming operation
- Exceeding any budget → HUMAN_GATE
- Budgets can be overridden per-task by explicit human configuration
