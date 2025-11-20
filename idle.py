"""
	idle.py - Q3 (DCU)
	by Ksuserkqy(20251113620)
	Docs: https://www.ksuser.cn/dcu/
	2025-10-20
"""

import time
import threading
from typing import Optional
from settings import IDLE_TIMEOUT_SECONDS, IDLE_CHECK_INTERVAL_SECONDS

class IdleManager:
    def __init__(self, model_getter, timeout_seconds:int=IDLE_TIMEOUT_SECONDS, check_interval:int=IDLE_CHECK_INTERVAL_SECONDS):
        self.model_getter = model_getter
        self.timeout_seconds = timeout_seconds
        self.check_interval = check_interval
        self._last_activity = time.time()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def touch(self) -> None:
        with self._lock:
            self._last_activity = time.time()

    def last_activity(self) -> float:
        with self._lock:
            return self._last_activity

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                now = time.time()
                last = self.last_activity()
                idle = now - last
                if idle >= self.timeout_seconds:
                    model = self.model_getter()
                    try:
                        if hasattr(model, "is_sleeping") and not model.is_sleeping():
                            print("[KTSE] 当前模型长时间处于空闲状态，已进入睡眠模式")
                            model.sleep()
                    except Exception:
                        pass
                    self.touch()
                time.sleep(self.check_interval)
            except Exception:
                # 保持运行，不让后台线程因为偶发错误退出
                time.sleep(self.check_interval)
