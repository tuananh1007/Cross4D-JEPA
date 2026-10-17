import logging
import sys


def get_logger(name: str = "cross4d_jepa", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    h = logging.StreamHandler(sys.stdout)  # stdout so it streams in SLURM logs
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    logger.propagate = False
    return logger

