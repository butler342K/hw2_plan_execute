"""Логування повної траєкторії виконання агента у JSON-файл."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class TrajectoryLogger:
    path: Path = field(default_factory=lambda: Path('trajectory.json'))
    events: List[Dict[str, Any]] = field(default_factory=list)
    start_time: float = field(default_factory=time.monotonic)

    def log(self, step: str, data: Dict[str, Any]) -> None:
        self.events.append({
            'step': step,
            'data': data,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'elapsed_ms': int((time.monotonic() - self.start_time) * 1000),
        })

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        target = Path(path) if path is not None else self.path
        target.write_text(
            json.dumps({
                'total_steps': len(self.events),
                'total_time_ms': int((time.monotonic() - self.start_time) * 1000),
                'trajectory': self.events,
            }, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
