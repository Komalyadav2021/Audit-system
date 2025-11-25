# src/utils/logger.py
import logging
from logging.handlers import RotatingFileHandler
import os

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("audit_ai")
logger.setLevel(logging.DEBUG)

fh = RotatingFileHandler(os.path.join(LOG_DIR, "audit_ai.log"), maxBytes=5_000_000, backupCount=3)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

# also print to console in dev
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)
