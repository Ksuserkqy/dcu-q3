"""
	settings.py - Q3 (DCU)
	by Ksuserkqy(20251113620)
	Docs: https://www.ksuser.cn/dcu/
	2025-10-20
"""

# Basic
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MODEL_DIR = f"./models/{MODEL_ID}"
DEFAULT_MAX_NEW_TOKENS = 50
DEFAULT_PROMPT = "The future of artificial intelligence is"

# Advanced
TTFT_TOKENS = 1
ASSERT_TTFT = False # 当 TTFT大于5时是否直接退出进程
KEEP_LAYERS = 0 # 睡眠时保留在 GPU 上的层数，0表示全部迁移到 CPU
IDLE_TIMEOUT_SECONDS = 300 # 超过多少秒无活动后触发模型 sleep（默认 300s = 5min）
IDLE_CHECK_INTERVAL_SECONDS = 5 # 后台检查间隔（秒），建议不要太小以免频繁轮询