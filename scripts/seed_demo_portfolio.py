"""Run from repo root: python scripts/seed_demo_portfolio.py"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
runpy.run_path(str(ROOT / "scripts" / "seed_demo_portfolio.py"), run_name="__main__")
