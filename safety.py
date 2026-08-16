"""Захисні механізми для ReAct-агента: max_steps, timeout, детекція зациклення."""

import json
import time
from typing import Any, Dict, List, Optional, Tuple

MAX_STEPS = 10
TIMEOUT_SECONDS = 60
MAX_REPEATS = 3


class LoopDetector:
    """Виявляє зациклення: N однакових поспіль викликів tool з тими самими аргументами."""

    def __init__(self, max_repeats: int = MAX_REPEATS):
        self.max_repeats = max_repeats
        self.recent_calls: List[Tuple[str, str]] = []

    def check(self, tool_name: str, args: Dict[str, Any]) -> bool:
        """Зареєструвати виклик tool і перевірити, чи останні max_repeats викликів однакові."""
        signature = (tool_name, json.dumps(args, sort_keys=True, ensure_ascii=False))
        self.recent_calls.append(signature)
        last_n = self.recent_calls[-self.max_repeats:]
        return len(last_n) == self.max_repeats and len(set(last_n)) == 1


class RunGuard:
    """Об'єднує max_steps, timeout і LoopDetector для одного запуску агента.

    Живе в AgentState (не через reducer — просто передається як об'єкт),
    тому створюється заново на кожен app.invoke() і не протікає між запусками.
    """

    def __init__(
        self,
        max_steps: int = MAX_STEPS,
        timeout_seconds: float = TIMEOUT_SECONDS,
        max_repeats: int = MAX_REPEATS,
    ):
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds
        self.loop_detector = LoopDetector(max_repeats=max_repeats)
        self.start_time = time.monotonic()

    def check_step_limit(self, step: int) -> Optional[str]:
        if step >= self.max_steps:
            return f'Досягнуто ліміту кроків ({self.max_steps}).'
        return None

    def check_timeout(self) -> Optional[str]:
        elapsed = time.monotonic() - self.start_time
        if elapsed > self.timeout_seconds:
            return f'Перевищено тайм-аут виконання ({self.timeout_seconds:.0f} с).'
        return None

    def check_loop(self, tool_calls: List[Dict[str, Any]]) -> Optional[str]:
        for call in tool_calls:
            if self.loop_detector.check(call.get('name', ''), call.get('args', {}) or {}):
                return (
                    f'Виявлено зациклення: tool "{call.get("name")}" '
                    f'викликається повторно з тими самими аргументами.'
                )
        return None
