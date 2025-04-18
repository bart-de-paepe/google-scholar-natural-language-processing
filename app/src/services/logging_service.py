import logging
import os
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

LOGGING_FILENAME = os.getenv("LOGGING_FILENAME")
LOGGING_LEVEL = os.getenv("LOGGING_LEVEL")
class LoggingService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        filename = os.path.join(str(Path(__file__).parent.parent.parent.parent), 'log', LOGGING_FILENAME)
        if LOGGING_LEVEL == 'DEBUG':
            logging.basicConfig(handlers=[TimedRotatingFileHandler(filename, when='D', interval=1)], encoding='utf-8',
                                level=logging.DEBUG)
        if LOGGING_LEVEL == 'INFO':
            logging.basicConfig(handlers=[TimedRotatingFileHandler(filename, when='D', interval=1)], encoding='utf-8',
                                level=logging.INFO)
        if LOGGING_LEVEL == 'WARNING':
            logging.basicConfig(handlers=[TimedRotatingFileHandler(filename, when='D', interval=1)], encoding='utf-8',
                                level=logging.WARNING)
        if LOGGING_LEVEL == 'ERROR':
            logging.basicConfig(handlers=[TimedRotatingFileHandler(filename, when='D', interval=1)], encoding='utf-8',
                                level=logging.ERROR)
