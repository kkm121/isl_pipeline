# Human Gate — Mandatory Escalation Rule

The autonomous pipeline MUST escalate to human intervention (HUMAN_GATE state) under these conditions:

## Mandatory Escalation Triggers
1. **Uncertain diagnosis**: The diagnostic engine cannot confidently classify the failure (confidence < 0.7)
2. **Destructive operation**: Any operation that would delete data, drop tables, remove checkpoints, or alter production state
3. **Repeated failure**: The same failure signature has occurred 2 or more times consecutively
4. **Unsafe remediation**: The proposed fix involves changing core model architecture, removing safety checks, or disabling validation
5. **Credential issue**: Any error involving authentication, authorization, or credential management
6. **Policy/ToS violation**: Any action that may violate platform terms of service (e.g., automated account creation, aggressive rate limiting circumvention)
7. **Max retries exceeded**: The retry policy limits have been reached (default: 2 per failure type, 5 total)
8. **Resource budget exceeded**: Any resource budget limit has been hit

## Behavior at HUMAN_GATE
- The pipeline STOPS completely
- A structured report is generated with: what was attempted, why it failed, what escalated it, and what options exist
- NO further autonomous action is taken until a human explicitly approves or overrides
- The state machine enforces this mechanically — the LLM cannot bypass HUMAN_GATE

## The LLM is NOT the state machine
The deterministic state machine controller (`src/orchestrator/state_machine.py`) enforces all transitions. The LLM provides evidence and decisions, but the transition logic is hardcoded. The LLM cannot:
- Skip verification steps
- Bypass retry limits
- Override HUMAN_GATE
- Self-approve its own work
