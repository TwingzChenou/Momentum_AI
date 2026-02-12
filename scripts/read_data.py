import os
import sys
from dotenv import load_dotenv

# Force Spark to use the Python version of the current environment
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Add project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from src.common.logging_utils import setup_logging
from src.common.setup_spark import create_spark_session

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
BUCKET_NAME = os.getenv("BUCKET_NAME")
LAKE_PATH = 

if not BUCKET_NAME:
    raise ValueError("BUCKET_NAME environment variable is not set")

HISTORY_LAKE_PATH = f"gs://{BUCKET_NAME}/silver/sp500_composition_history"