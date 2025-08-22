import logging
import os
import sys
from config.settings import LOG_DIR, TIMESTAMP

def setup_logger(name, log_file=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

log_file = os.path.join(LOG_DIR, f'training_{TIMESTAMP}.log')
logger = setup_logger('face_recognition', log_file)

progress_logger = setup_logger(
    'progress', 
    os.path.join(LOG_DIR, f'progress_{TIMESTAMP}.log')
)