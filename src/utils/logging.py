import logging
import json
import sys
from datetime import datetime
from pathlib import Path

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage()
        }
        if hasattr(record, 'extra_fields'):
            log_obj.update(record.extra_fields)
        return json.dumps(log_obj)

def setup_logging(log_dir: str = 'logs/local', level: str = 'INFO', experiment_name: str = 'default') -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    ch.setFormatter(console_formatter)
    logger.addHandler(ch)
    
    # File handler
    log_file = Path(log_dir) / f"{experiment_name}.log"
    fh = logging.FileHandler(str(log_file))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(console_formatter)
    logger.addHandler(fh)
    
    # JSON File handler
    json_file = Path(log_dir) / f"{experiment_name}.jsonl"
    jh = logging.FileHandler(str(json_file))
    jh.setLevel(logging.DEBUG)
    jh.setFormatter(JSONFormatter())
    logger.addHandler(jh)
    
    return logger
