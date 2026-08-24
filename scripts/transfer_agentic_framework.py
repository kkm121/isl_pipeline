"""
=============================================================================
Script: Transfer & Scaffold Agentic Coding Framework to Downloads/vernacular
=============================================================================
Transfers:
  1. .agents/ (Master Swarm Architecture, 9 Core Agent Skills, 7 Governance Rules)
  2. MCP configs, hooks, safety/formatting scripts
  3. Domain-agnostic scaffolding (README.md, pyproject.toml, requirements.txt, src/, tests/)
=============================================================================
"""

import os
import shutil
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent.parent
DEST_DIR = Path(r"C:\Users\muthu\Downloads\vernacular")

def transfer():
    print(f"Transferring Agentic Coding Framework:")
    print(f"  Source: {SOURCE_DIR}")
    print(f"  Destination: {DEST_DIR}")
    
    os.makedirs(DEST_DIR, exist_ok=True)
    os.makedirs(DEST_DIR / ".agents", exist_ok=True)
    os.makedirs(DEST_DIR / ".agents" / "skills", exist_ok=True)
    os.makedirs(DEST_DIR / ".agents" / "rules", exist_ok=True)
    os.makedirs(DEST_DIR / ".agents" / "scripts", exist_ok=True)
    os.makedirs(DEST_DIR / "src", exist_ok=True)
    os.makedirs(DEST_DIR / "tests", exist_ok=True)
    os.makedirs(DEST_DIR / "data", exist_ok=True)
    os.makedirs(DEST_DIR / "outputs", exist_ok=True)

    # 1. Transfer Rules
    src_rules = SOURCE_DIR / ".agents" / "rules"
    if src_rules.exists():
        for rule_file in src_rules.glob("*.md"):
            shutil.copy2(rule_file, DEST_DIR / ".agents" / "rules" / rule_file.name)
            print(f"  [Rule] Copied {rule_file.name}")

    # 2. Transfer Core Agent Skills
    src_skills = SOURCE_DIR / ".agents" / "skills"
    if src_skills.exists():
        for skill_dir in src_skills.iterdir():
            if skill_dir.is_dir():
                dest_skill_dir = DEST_DIR / ".agents" / "skills" / skill_dir.name
                os.makedirs(dest_skill_dir, exist_ok=True)
                for skill_file in skill_dir.glob("*"):
                    shutil.copy2(skill_file, dest_skill_dir / skill_file.name)
                print(f"  [Skill] Copied {skill_dir.name}")

    # 3. Transfer Hooks, MCP Config, Scripts
    for fname in ["hooks.json", "mcp_config.json"]:
        p = SOURCE_DIR / ".agents" / fname
        if p.exists():
            shutil.copy2(p, DEST_DIR / ".agents" / fname)
            print(f"  [Config] Copied {fname}")

    src_scripts = SOURCE_DIR / ".agents" / "scripts"
    if src_scripts.exists():
        for script_file in src_scripts.glob("*.py"):
            shutil.copy2(script_file, DEST_DIR / ".agents" / "scripts" / script_file.name)
            print(f"  [Script] Copied {script_file.name}")

    # 4. Create Master AGENTS.md for the Vernacular Project Setup
    agents_md_content = """# Autonomous Multi-Agent Engineering Swarm Architecture
**Evidence-Based, Gate-Enforced Engineering Framework**

---

## 🏛️ 4-Plane Engineering Architecture

1. **Control Plane** — High-level reasoning, physical/domain invariance enforcement, multi-agent task dispatch.
2. **Integration Plane** — Fast MCP protocols for dataset streaming, linter execution, and GPU/model dispatch.
3. **Execution Plane** — Sealed containers (`--network=none --read-only`), local runtime, and remote GPU compute.
4. **Verification Plane** — Deterministic gate enforcement with strict mathematical, physical, and quantitative proofs.

---

## 👥 The Core Specialized Subagent Roles

| # | Agent Name | Primary Model Tier | Reasoning Level | Core Domain & Specialization | Tool Scope |
|:---|:---|:---|:---|:---|:---|
| 1 | **`principal_architect`** | **Claude Opus / Gemini Pro** | **High** | End-to-end architecture orchestration, specification compliance, roadmap & gate execution | Orchestration, planning, git, read tools |
| 2 | **`critic` / `adversarial_critic`** | **Claude Opus / Gemini Pro** | **High** | Data leakage detection, numerical stability audit, anti-hallucination & anti-fabrication enforcement | Read-only tools, AST analyzer, diff inspector |
| 3 | **`independent_reviewer`** | **Claude Opus / Gemini Pro** | **High** | Clean-session diff verification, mathematical derivation checks, gate sign-off | Read-only tools, diff inspector |
| 4 | **`code_writer`** | **Gemini Pro / Flash** | **Medium-High** | Precision TDD implementation, modular architecture, 100% strict type annotations | `src/`, file editing tools |
| 5 | **`test_engineer`** | **Gemini Pro / Flash** | **Medium-High** | Comprehensive unit, integration, property-based, and edge-case testing (`pytest`) | `tests/`, test runners |
| 6 | **`benchmark`** | **Gemini Flash** | **Medium** | Quantitative performance metrics, latency/throughput profiling, convergence, regression tracking | `src/evaluation/`, metrics |
| 7 | **`researcher`** | **Gemini Flash** | **Medium** | Repository exploration, paper analysis, API/dataset documentation research | Search, read-only tools |
| 8 | **`ml_ops`** | **Gemini Pro / Flash** | **Medium** | Model training orchestration, remote GPU dispatch (Kaggle/Cloud), ONNX/INT8 quantization, export | Remote execution, scripts |
| 9 | **`verify_agent`** | **Gemini Flash** | **Medium** | Deterministic static and dynamic verification, lint enforcement (`ruff`, `mypy`), return code asserting | Terminal, test runners |

---

## 🔒 The 6 Engineering Gates (Before Deployment)

1. **Gate 1 (Specification & Requirements Gate)**: Machine-verifiable acceptance criteria approved before writing implementation code.
2. **Gate 2 (Data Provenance & Integrity Gate)**: Zero synthetic fabrication where real data is expected; verified train/val/test splits.
3. **Gate 3 (Test-Driven Baseline Gate)**: 100% test pass rate with regression assertions against established baselines.
4. **Gate 4 (Numerical Stability & Edge Case Audit)**: Zero NaNs/Infs under all boundary conditions and precision tiers.
5. **Gate 5 (Adversarial Critic Sign-Off)**: Independent verification of claims with raw terminal output and zero subjective hand-waving.
6. **Gate 6 (Final Deployment & Calibration Gate)**: Latency, memory, and calibration validation on target runtime hardware.
"""
    with open(DEST_DIR / ".agents" / "AGENTS.md", "w", encoding="utf-8") as f:
        f.write(agents_md_content)
    print("  [Master] Created .agents/AGENTS.md")

    # 5. Create Root README.md
    readme_content = """# Autonomous Agentic Coding Workspace

This workspace is equipped with the **Evidence-Based, Multi-Agent Swarm Framework**.

---

## 📁 Workspace Structure

```
├── .agents/
│   ├── AGENTS.md            # Master 4-plane multi-agent swarm architecture & gates
│   ├── hooks.json           # Lifecycle event hooks (pre-commit, post-test)
│   ├── mcp_config.json      # Model Context Protocol server configuration
│   ├── rules/               # 7 strict governance rules (evidence-based, gates, safety)
│   │   ├── evidence_based.md
│   │   ├── human_gate.md
│   │   ├── no_synthetic_data.md
│   │   ├── resource_budgets.md
│   │   ├── retry_limits.md
│   │   ├── security_boundaries.md
│   │   └── specification_gate.md
│   ├── scripts/             # Safety check & auto-formatting hook scripts
│   │   ├── auto_format.py
│   │   └── safety_check.py
│   └── skills/              # 9 specialized agent skill definitions
│       ├── benchmark/
│       ├── code_writer/
│       ├── critic/
│       ├── ml_ops/
│       ├── principal_engineer/
│       ├── researcher/
│       ├── reviewer/
│       ├── test_engineer/
│       └── verify_agent/
├── data/                    # Datasets & raw inputs
├── outputs/                 # Artifacts, benchmark reports, evaluation metrics
├── src/                     # Core application source code
└── tests/                   # Automated pytest test suite
```

---

## 🚀 How to Use

1. **Activate the Swarm**: Point Antigravity / Gemini to this directory. The `.agents/` configuration and skills will automatically activate.
2. **Execute Tasks with Strict Gates**:
   - The **Principal Engineer** will oversee planning and gate enforcement.
   - The **Critic Agent** will audit all claims with raw numbers and diffs.
   - The **Test Engineer** and **Code Writer** will build features using Test-Driven Development (TDD).
3. **Run Tests**:
   ```bash
   pytest tests/ -v
   ```
"""
    with open(DEST_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("  [Doc] Created README.md")

    # 6. Create pyproject.toml
    pyproject_content = """[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "vernacular-ai-suite"
version = "1.0.0"
description = "Autonomous Agentic AI Suite"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0.0",
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "pydantic>=2.0.0",
    "fastapi>=0.100.0",
    "uvicorn>=0.22.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "mypy>=1.5.0",
    "ruff>=0.0.280",
    "black>=23.7.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]

[tool.ruff]
line-length = 100
target-version = "py310"
"""
    with open(DEST_DIR / "pyproject.toml", "w", encoding="utf-8") as f:
        f.write(pyproject_content)
    print("  [Build] Created pyproject.toml")

    # 7. Create starter files in src and tests
    (DEST_DIR / "src" / "__init__.py").touch()
    (DEST_DIR / "tests" / "__init__.py").touch()

    # Create a simple verification test in tests/
    test_smoke_content = """def test_agentic_environment_ready():
    assert True
"""
    with open(DEST_DIR / "tests" / "test_smoke.py", "w", encoding="utf-8") as f:
        f.write(test_smoke_content)
    print("  [Test] Created tests/test_smoke.py")

    print("\n[SUCCESS] Agentic coding setup successfully transferred to Downloads/vernacular!")

if __name__ == "__main__":
    transfer()
