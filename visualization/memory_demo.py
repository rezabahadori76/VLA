from __future__ import annotations

import json
from pathlib import Path
from typing import List


def export_memory_demo(memory, output_path: Path, queries: List[str]) -> None:
    payload = {q: memory.query(q) for q in queries}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
