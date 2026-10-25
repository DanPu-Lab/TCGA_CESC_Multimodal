from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.engine import training_main

if __name__ == "__main__":
    training_main("late_fusion", "final")
