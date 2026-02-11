import logging
import sys
from pathlib import Path
from loguru import logger

# On définit où stocker les logs
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

def setup_logging():
    # On nettoie les loggers par défaut de Python qui sont moches
    logging.root.handlers = [InterceptHandler()]
    logging.root.setLevel(logging.INFO)

    # On supprime les configurations précédentes
    logger.remove()

    # 1. Sortie CONSOLE (Jolies couleurs pour le dév)
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    # 2. Sortie FICHIER (Pour l'historique des erreurs)
    # "rotation" crée un nouveau fichier chaque jour ou quand il fait 10 Mo
    logger.add(
        LOG_FILE,
        level="ERROR",  # On n'écrit que les erreurs et warnings dans le fichier pour pas le bourrer
        rotation="10 MB",
        retention="10 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )

# Petite classe utilitaire pour rediriger les logs standards vers Loguru
class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())