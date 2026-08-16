# Specification Gate

This rule enforces Section 2 of the architecture: The Specification Gate.

Before ANY implementation work begins, the following steps must be completed and approved:

1. **Research**: Research the objective thoroughly.
2. **Specification**: Write a precise specification document outlining the required changes.
3. **Acceptance Criteria**: Define granular, programmatically verifiable acceptance criteria.
4. **Test First**: Define and write tests BEFORE writing the actual implementation (TDD).
5. **Approval**: Obtain explicit approval from the specification gate.

### Examples

**Bad Example:**
> "Improve the training model"
*Why it's bad: Vague, unmeasurable, lacks specific tests or thresholds.*

**Good Example:**
> "Implement attention mechanism in classifier, preserve predict() API signature, add 3 regression tests, pass pytest, achieve val_accuracy >= 0.85, produce checkpoint at checkpoints/attn_v1.pt"
*Why it's good: Precise, API-aware, includes test requirements, defines a specific quantitative metric, and names an expected artifact.*

### Verification Requirement
All acceptance criteria must be machine-verifiable. Acceptable forms of verification include:
- Test pass/fail counts
- Metric thresholds (e.g., accuracy > X)
- File existence and size (e.g., artifact creation)
