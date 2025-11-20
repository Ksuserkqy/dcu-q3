"""
	ktse.py - Q3 (DCU)
	by Ksuserkqy(20251113620)
	Docs: https://www.ksuser.cn/dcu/
	2025-10-21
"""

import gc
import time
# import torch
import threading
from tools import *
from accelerate import dispatch_model
from transformers import TextIteratorStreamer
from accelerate.hooks import remove_hook_from_submodules
from transformers import AutoModelForCausalLM, AutoTokenizer

class KsuserTransformersSleepEngine:
	def __init__(self, 
		model_dir: str, 
		trust_remote_code: bool = True, 
		keep_layers: int = 0
	) -> None:
		self.model_dir = model_dir
		self.trust_remote_code = trust_remote_code
		self._sleeping: bool = False
		# 记录最后一次活动时间，用于外部 IdleManager 或者本地查询
		try:
			self._last_activity = time.time()
		except Exception:
			self._last_activity = 0.0

		# 允许“部分睡眠”保留前 N 层与嵌入/输出头在 GPU（可通过 keep_layers 修改）
		self.keep_layers = max(0, int(keep_layers or 0))

		backend = get_device_backend()
		self.dtype = torch.float16 if backend in ("cuda", "mps") else torch.float16

		# 加载分词器与模型
		self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=trust_remote_code)
		self.model = AutoModelForCausalLM.from_pretrained(
			self.model_dir,
			dtype=self.dtype,
			trust_remote_code=trust_remote_code,
			device_map="auto",
		)
		self.model.eval()

		self.original_device_map = "auto"

	def _clear_cuda_cache(self) -> None:
		if torch.cuda.is_available():
			try:
				for i in range(torch.cuda.device_count()):
					with torch.cuda.device(i):
						torch.cuda.empty_cache()
			except Exception:
				pass

			# 进一步回收 Driver 侧碎片
			try:
				torch.cuda.ipc_collect()
			except Exception:
				pass

			# 确保统计前释放已完成
			try:
				torch.cuda.synchronize()
			except Exception:
				pass

	def _synchronize_all_cuda(self) -> None:
		if torch.cuda.is_available():
			try:
				for i in range(torch.cuda.device_count()):
					with torch.cuda.device(i):
						torch.cuda.synchronize()
			except Exception:
				pass

	def sleep(self) -> None:
		"""睡眠模式，启动！"""

		# 先移除可能存在的 accelerate hooks，避免设备路由干扰
		remove_hook_from_submodules(self.model)

		# 迁移权重到 CPU（覆盖原有 device_map 分片）
		try:
			if isinstance(self.original_device_map, dict) and self.keep_layers > 0:
				# 构造“部分睡眠”的 device_map：保留前 N 层 + 嵌入 + lm_head 在原 GPU，其余映射到 CPU
				cpu_map = {}
				def keep_key(name: str) -> bool:
					if name.startswith("model.embed_tokens") or name.startswith("transformer.wte"):
						return True
					if name.startswith("lm_head"):
						return True
					# 前 N 层
					for i in range(self.keep_layers):
						if name.startswith(f"model.layers.{i}.") or name.startswith(f"transformer.h.{i}."):
							return True
					return False
				for k, v in self.original_device_map.items():
					if keep_key(k):
						cpu_map[k] = v  # 保持在原 GPU
					else:
						cpu_map[k] = "cpu"
				self.model = dispatch_model(self.model, device_map=cpu_map)
			else:
				self.model = dispatch_model(self.model, device_map="cpu")
		except Exception:
			self.model = self.model.to("cpu")

		# 尝试将 CPU 上的参数与缓冲区固定为 pinned memory，以加速 H2D 传输
		try:
			for module in self.model.modules():
				for name, param in list(module._parameters.items()) if hasattr(module, "_parameters") and module._parameters else []:
					if param is not None and not param.is_cuda and hasattr(param, "pin_memory"):
						try:
							module._parameters[name] = param.pin_memory()
						except Exception:
							pass
				for name, buf in list(module._buffers.items()) if hasattr(module, "_buffers") and module._buffers else []:
					if buf is not None and not getattr(buf, "is_cuda", False) and hasattr(buf, "pin_memory"):
						try:
							module._buffers[name] = buf.pin_memory()
						except Exception:
							pass
		except Exception:
			pass
		for _ in range(3):
			torch.cuda.empty_cache()
		self._sleeping = True
		# 更新活动时间，避免被 IdleManager 立即再次触发
		try:
			self._last_activity = time.time()
		except Exception:
			pass

		# 确保所有 GPU 操作完成后再清理缓存并同步，保证显存观察到真实释放
		try:
			gc.collect()
		except Exception:
			pass
		self._synchronize_all_cuda()
		self._clear_cuda_cache()

	def wake(self) -> None:
		"""将模型权重从 CPU 恢复到原始 device_map"""
		try:
			# self.model = dispatch_model(self.model, device_map="auto")  # 被 ["a", "u", "t", "o"] 折磨疯啦，索性直接固定auto乐
			self.model = dispatch_model(self.model, device_map="auto")  # auto还是爆显存，干脆第一张卡多给点空间用来生成
			self._sleeping = False
			return
		except Exception as error:
			pass
		# 回退：若无法分片分配，则放到单卡/默认设备
		backend = get_device_backend()
		device = torch.device("cuda") if backend == "cuda" else torch.device("mps")
		self.model = self.model.to(device)
		self._sleeping = False
		# 更新活动时间
		try:
			self._last_activity = time.time()
		except Exception:
			pass
		# 同步，避免后续计时受到异步 H2D 影响
		self._synchronize_all_cuda()

	@torch.inference_mode()
	def generate_prepared(self, inputs: dict[str, torch.Tensor], max_new_tokens: int) -> str:
		"""避免重复分词带来的额外延迟，inputs 需为 CPU Tensor（会将其放到首个 CUDA 设备（若可用））"""
		backend = get_device_backend()
		inp = inputs
		if backend == "cuda" and torch.cuda.is_available():
			inp = {k: v.to("cuda:0") for k, v in inputs.items()}
		elif backend == "mps":
			inp = {k: v.to("mps") for k, v in inputs.items()}
		# 注释部分功能为输出时不输出prompt，仅生成新 tokens
		# outputs = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
		# input_len = inp["input_ids"].shape[1]
		# gen_only_ids = outputs[0][input_len:]
		# return self.tokenizer.decode(gen_only_ids, skip_special_tokens=True)
		outputs = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
		# 记录当前活动时刻
		try:
			self._last_activity = time.time()
		except Exception:
			pass
		return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

	@torch.inference_mode()
	def generate(self, prompt: str, max_new_tokens: int) -> str:
		inputs = self.tokenizer(prompt, return_tensors="pt")
		# 将输入放到第一个可用设备（若为分片模型，Hf hooks 会自动路由）
		backend = get_device_backend()
		if backend == "cuda" and torch.cuda.is_available():
			inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
		elif backend == "mps":
			inputs = {k: v.to("mps") for k, v in inputs.items()}
		outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
		# 记录当前活动时刻
		try:
			self._last_activity = time.time()
		except Exception:
			pass
		return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
	
	@torch.inference_mode()
	def generate_stream(self, prompt: str, max_new_tokens: int):
		"""流式生成"""
		inputs = self.tokenizer(prompt, return_tensors="pt")
		backend = get_device_backend()
		if backend == "cuda" and torch.cuda.is_available():
			inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
		elif backend == "mps":
			inputs = {k: v.to("mps") for k, v in inputs.items()}
	
		streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
		
		# Async
		thread = threading.Thread(target=self.model.generate, kwargs={
			"input_ids": inputs["input_ids"],
			"attention_mask": inputs.get("attention_mask", None),
			"max_new_tokens": max_new_tokens,
			"do_sample": False,
			"streamer": streamer
		})
		thread.start()

		for new_text in streamer:
			try:
				self._last_activity = time.time()
			except Exception:
				pass
			yield new_text
		thread.join()

	def touch_activity(self) -> None:
		"""供外部调用，标记模型有活动（更新时间戳）。"""
		try:
			self._last_activity = time.time()
		except Exception:
			pass

	def is_sleeping(self) -> bool:
		return bool(self._sleeping)

if __name__ == "__main__":
	KsuserTransformersSleepEngine("8")
	print("KsuserTransformersSleepEngine(KTSE) module loaded.")