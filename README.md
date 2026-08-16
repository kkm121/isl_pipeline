# ISL Pipeline — Autonomous ML Engineering Architecture

This project implements an Indian Sign Language (ISL) recognition pipeline, orchestrated through a robust, autonomous Machine Learning Engineering architecture.

## Architecture Overview

The system is structured into 4 distinct planes to ensure reliability, security, and scalability:

1. **Control Plane**: Governs reasoning, planning, and delegation.
2. **Integration Plane**: Manages external connections, credentials, and deployment environments (like Kaggle).
3. **Execution Plane**: Handles actual ML training, data processing, and inference.
4. **Verification Plane**: Enforces evidence-based validation (tests, static analysis, metrics) before accepting any change.

## Agent Roles

The architecture employs specialized agents, each with specific mandates:

| Role | Responsibility |
|------|----------------|
| **Principal Engineer** | Reasons, plans, delegates, evaluates evidence, makes final decisions. |
| **Researcher** | Investigates codebase, literature, documentation, and issues. |
| **Test Engineer** | Writes tests, runs CI (pytest, mypy, ruff), and analyzes failures. |
| **Reviewer** | Conducts adversarial independent code review, searching for flaws. |
| **ML-Ops** | Manages remote training (Kaggle), monitors status, and performs diagnostics. |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/isl_pipeline.git
cd isl_pipeline

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with dev dependencies
pip install -e .[dev]
```

### Running Tests

```bash
pytest tests/
```

### Training

```bash
# Example local training run
python -m src.training.train --config config/default.yaml
```

## MCP Servers

The project utilizes 4 core Model Context Protocol (MCP) servers:
1. **Repository Explorer**: For codebase analysis and navigation.
2. **Linter/Test Runner**: For executing pytest, mypy, ruff, and gathering evidence.
3. **Kaggle Manager**: For submitting, monitoring, and retrieving remote kernel runs.
4. **Credential Broker**: For secure, isolated management of API keys and tokens.

## Kaggle Integration

ML-Ops agents utilize the Kaggle Manager MCP to push training kernels to Kaggle's remote GPU environment, safely retrieving logs and model weights upon completion.

## Docker

- **Sealed Sandbox**: Used for executing generated code locally without network access to prevent unauthorized external connections.
- **Controlled Integration**: Specialized containers that allowlist necessary endpoints (kaggle.com, github.com, pypi.org) for specific CI/CD and deployment tasks.

## Directory Structure Overview

- `src/`: Core implementation (data, model, training, inference).
- `tests/`: Unit and integration tests.
- `config/`: Configuration files for models and pipelines.
- `notebooks/`: Exploratory data analysis and experimental kernels.
- `.agents/`: Project-specific agent rules and skills definitions.

## Security Boundaries

1. **Model Boundary**: Agents access tools and outputs, never raw credentials.
2. **Filesystem Boundary**: Agents are restricted to the project workspace.
3. **Network Boundary**: Zero network access during test execution; controlled allowlists for integration tasks.
4. **Credential Boundary**: All secrets flow through a dedicated broker, strictly isolated from logs and code.

## License

MIT
