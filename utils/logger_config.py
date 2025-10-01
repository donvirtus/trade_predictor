import logging, os, sys
from datetime import datetime


def get_logger(name: str = "main", log_dir: str = "data/logs", level: int = logging.INFO) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)

    file_path = os.path.join(log_dir, f"{name}_{datetime.utcnow().strftime('%Y%m%d')}.log")
    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.debug("Logger initialized at %s", file_path)
    return logger
