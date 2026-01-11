"""Configuration settings for the two-agent system"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """Application settings"""

    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Model Configuration
    MODEL_NAME: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    MODEL_PROVIDER: str = os.getenv("MODEL_PROVIDER", "openai")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))
    TOP_P: float = float(os.getenv("TOP_P", "0.1"))

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    PARTIAL_DIR: Path = BASE_DIR / "artifact/intermediate"
    RESULT_DIR: Path = BASE_DIR / "artifact/result"
    RAW_DATA_DIR: Path = BASE_DIR / "input/raw_data"
    LOG_DIR: Path = BASE_DIR / "logs"

    # Input
    INPUT_FILENAME: str = os.getenv("INPUT_FILENAME", "10k_red_data_v1.0.xlsx")
    INPUT_NAME: Path = RAW_DATA_DIR / INPUT_FILENAME
    # Input Text Column Name
    TEXT_COL: str = os.getenv("TEXT_COL", "title_body")
    # Filter out entities with frequency below this treshold
    FREQ_UB: int = os.getenv("FREQ_UB", '20')

    # Output
    OUTPUT_FILENAME: str = os.getenv("OUTPUT_FILENAME", "KG标签体系_v2.0.txt")
    RESULT_NAME: Path = RESULT_DIR / OUTPUT_FILENAME

    def __init__(self):
        """Initialize settings and create necessary directories"""
        self.PARTIAL_DIR.mkdir(exist_ok=True)
        self.RESULT_DIR.mkdir(exist_ok=True)

        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

settings = Settings()

