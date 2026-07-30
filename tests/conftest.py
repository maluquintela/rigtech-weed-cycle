"""Ajustes de import: garante que ``import src...`` funcione ao rodar ``pytest``
a partir da raiz do repositório."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
