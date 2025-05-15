import logging
from logging.handlers import RotatingFileHandler
import os

class ColorFormatter(logging.Formatter):
    COLOR_CODES = {
        logging.DEBUG: '\033[36m',    # Cyan
        logging.INFO: '\033[32m',     # Green
        logging.WARNING: '\033[33m',  # Yellow
        logging.ERROR: '\033[31m',     # Red
        logging.CRITICAL: '\033[31;1m' # Bold Red
    }
    RESET_CODE = '\033[0m'
    
    def format(self, record):
        filename = os.path.basename(record.pathname)
        func_info = f":{record.funcName}()" if record.funcName and record.funcName != "<module>" else ""
        
        color = self.COLOR_CODES.get(record.levelno, '')
        message = super().format(record)
        return f"{color}{filename}{func_info} | {message}{self.RESET_CODE}"

def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s | %(levelname)-8s | %(filename)s:%(funcName)-15s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            RotatingFileHandler(
                'app.log',
                maxBytes=1024*1024,
                backupCount=3
            )
        ]
    )

    # Apply color formatter to console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColorFormatter())
    logging.getLogger().addHandler(console_handler)

    # Reduce uvicorn noise
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)