import sys
from pathlib import Path

# Make the project root importable so tests can do `from src.llm_client_0 import ...`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
