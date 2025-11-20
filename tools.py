"""
	tools.py - Q3 (DCU)
	by Ksuserkqy(20251113620)
	Docs: https://www.ksuser.cn/dcu/
	2025-10-20
"""

import torch
from typing import Optional, Union


def get_device_backend() -> str:
	if torch.cuda.is_available():
		return "cuda"
	if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
		return "mps"
	raise SystemExit("当前环境未检测到 CUDA 或 MPS。该项目仅支持 CUDA/MPS，请在具备 NVIDIA GPU (HCU) 或 Apple M 系列设备的环境运行。")

def device_mem_get_info() -> Optional[tuple[int, int]]:
	"""返回 (free_bytes_sum, total_bytes_sum) 跨所有 GPU 的"""
	backend = get_device_backend()
	try:
		if backend == "cuda":
			n = torch.cuda.device_count()
			free_sum = 0
			total_sum = 0
			for i in range(n):
				with torch.cuda.device(i):
					free, total = torch.cuda.mem_get_info()
				free_sum += int(free)
				total_sum += int(total)
			return free_sum, total_sum
	except Exception:
		return None
	return None

def device_mem_get_info_per_gpu() -> Optional[dict[int, tuple[int, int]]]:
	"""返回 {device_index: (free_bytes, total_bytes)}；若不可用返回 None。"""
	backend = get_device_backend()
	try:
		if backend == "cuda" and hasattr(torch, "cuda") and torch.cuda.is_available():
			n = torch.cuda.device_count()
			out = {}
			for i in range(n):
				with torch.cuda.device(i):
					free, total = torch.cuda.mem_get_info()
				out[i] = (int(free), int(total))
			return out
	except Exception:
		return None
	return None

def format_gib(num_bytes: Optional[int]) -> Union[int, float]:
	if num_bytes is None:
		return 0
	return float(f"{num_bytes / (1024**3):.2f}")

def format_text_diff_preview(baseline: str, after: str) -> tuple[str, str]:
	return baseline[:120].replace("\n", " "), after[:120].replace("\n", " ")