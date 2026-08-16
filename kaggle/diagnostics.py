import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

class ErrorType(Enum):
    OOM = "OOM"
    NAN_GRADIENT = "NAN_GRADIENT"
    DEPENDENCY = "DEPENDENCY"
    CUDA = "CUDA"
    DATA = "DATA"
    TIMEOUT = "TIMEOUT"
    INFRASTRUCTURE = "INFRASTRUCTURE"
    LOGIC_MODEL = "LOGIC_MODEL"
    UNKNOWN = "UNKNOWN"

@dataclass
class DiagnosticReport:
    error_type: ErrorType
    root_cause: str
    recommended_action: str
    requires_human: bool
    confidence: float
    raw_error: str
    retryable: bool
    remediation_code: Optional[str] = None

class DiagnosticEngine:
    def __init__(self):
        self.patterns = {
            ErrorType.OOM: [r'CUDA out of memory', r'RuntimeError: CUDA error: out of memory', r'Killed'],
            ErrorType.NAN_GRADIENT: [r'nan', r'NaN', r'inf', r'exploding', r'gradient overflow'],
            ErrorType.DEPENDENCY: [r'ModuleNotFoundError', r'ImportError', r'No module named', r'version conflict'],
            ErrorType.CUDA: [r'CUDA error', r'cudnn', r'GPU not available', r'no CUDA GPUs'],
            ErrorType.DATA: [r'FileNotFoundError', r'No such file', r'corrupt', r'shape mismatch', r'KeyError'],
            ErrorType.TIMEOUT: [r'exceeded', r'timeout', r'Time limit'],
            ErrorType.INFRASTRUCTURE: [r'connection', r'network', r'HTTP', r'server error']
        }
        
        self.defaults = {
            ErrorType.OOM: ("Reduce batch size", True),
            ErrorType.NAN_GRADIENT: ("Lower learning rate", True),
            ErrorType.DEPENDENCY: ("Fix imports", False),
            ErrorType.CUDA: ("Check GPU setup", False),
            ErrorType.DATA: ("Check data path", False),
            ErrorType.TIMEOUT: ("Optimize code", False),
            ErrorType.INFRASTRUCTURE: ("Retry later", True),
            ErrorType.UNKNOWN: ("Review logs", True)
        }

    def diagnose(self, log_output: str, exit_code: int = 1) -> DiagnosticReport:
        for err_type, pats in self.patterns.items():
            for pat in pats:
                if re.search(pat, log_output):
                    action, retryable = self.defaults[err_type]
                    return DiagnosticReport(
                        error_type=err_type,
                        root_cause=f"Matched pattern {pat}",
                        recommended_action=action,
                        requires_human=not retryable,
                        confidence=0.9,
                        raw_error=log_output,
                        retryable=retryable
                    )
        
        return DiagnosticReport(
            error_type=ErrorType.UNKNOWN,
            root_cause="Unknown error",
            recommended_action=self.defaults[ErrorType.UNKNOWN][0],
            requires_human=True,
            confidence=0.1,
            raw_error=log_output,
            retryable=self.defaults[ErrorType.UNKNOWN][1]
        )

    def get_remediation(self, report: DiagnosticReport) -> Dict:
        if report.error_type == ErrorType.OOM:
            return {"config_changes": {"batch_size": "half", "gradient_checkpointing": True}}
        return {}
