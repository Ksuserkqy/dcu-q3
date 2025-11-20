"""
	main.py - Q3 (DCU)
	by Ksuserkqy(20251113620)
	Docs: https://www.ksuser.cn/dcu/
	2025-10-20
"""

import time
import ktse
from tools import *
from settings import *

def test(
	model_dir: str = MODEL_DIR,
	prompt: str = DEFAULT_PROMPT,
	max_new_tokens:int = DEFAULT_MAX_NEW_TOKENS,
	ttft_tokens: int = TTFT_TOKENS,
	assert_ttft: bool = ASSERT_TTFT,
	keep_layers: int = KEEP_LAYERS,
	) -> dict[str, object]:
	print(f"[KTSE] 模型将从 {model_dir} 加载")
	
	# 初始化
	engine = ktse.KsuserTransformersSleepEngine(model_dir=model_dir, keep_layers=keep_layers)

	backend = get_device_backend()
	info0 = device_mem_get_info()

	print(f"[KTSE] 当前设备被识别为 {backend}")
	if info0 is not None:
		free0, total0 = info0
		print(f"[KTSE] 当前显存占用情况 {(format_gib(total0)-format_gib(free0)):.2f} GiB / {format_gib(total0):.2f} GiB")
	else:
		print("[KTSE] 当前设备不支持显存信息读取")

	print("[KTSE] 基线生成中……")
	baseline_text = engine.generate(prompt, max_new_tokens)
	print(f"[KTSE] 基线输出：{baseline_text}")

	print("[KTSE] 进入睡眠模式 …")
	pre = device_mem_get_info()
	pre_used = (pre[1] - pre[0]) if pre else None

	engine.sleep()  # 进入睡眠

	# 等待短暂时间以稳定显存统计
	time.sleep(0.2)

	post = device_mem_get_info()
	free_after_sleep = post[0] if post else None
	release_bytes = None
	if pre is not None and post is not None:
		used_after_sleep = post[1] - post[0]
		release_bytes = max(0, pre_used - used_after_sleep) if pre_used is not None else None
		print(f"[KTSE] 入睡后占用: {(format_gib(post[1])-format_gib(post[0])):.2f} GiB / {format_gib(post[1]):.2f} GiB" + (f"  | 释放约: {format_gib(release_bytes):.2f} GiB" if release_bytes is not None else ""))
	else:
		print("[KTSE] 无法读取入睡后的显存信息")

	# TTFT：唤醒 + 生成 1 token（以 time.perf_counter 精准计时）
	print("[阶段] 测量 TTFT 中 ...")

	# 预分词，避免计时包含 tokenizer 开销
	prep_inputs = engine.tokenizer(prompt, return_tensors="pt")
	start = time.perf_counter()

	engine.wake() # 唤醒

	_ = engine.generate_prepared(prep_inputs, ttft_tokens)
	ttft_s = time.perf_counter() - start
	print(f"[KTSE] * 测量完成！ TTFT ≈ {ttft_s:.3f} s （阈值<5s）")
	if assert_ttft and ttft_s >= 5.0:
		raise SystemExit(f"TTFT {ttft_s:.3f}s ≥ 5s，未达标")

	print("[KTSE] 恢复后完整生成对比")
	text_after = engine.generate(prompt, max_new_tokens)
	print("[KTSE] 唤醒后输出：", text_after)
	same = (text_after == baseline_text)
	print(f"[KTSE] 与基线一致: {same}")
	if not same:
		b120, a120 = format_text_diff_preview(baseline_text, text_after)
		print("[KTSE] [差异-前120] 基线:", b120)
		print("[KTSE] [差异-前120] 现有:", a120)

	# 等待驱动层完成回迁占用统计
	time.sleep(0.2)
	info_after = device_mem_get_info()
	free_after_wake = None
	reload_bytes = None
	if info_after is not None and post is not None:
		free_after, total_after = info_after
		free_after_wake = free_after
		used_after = total_after - free_after
		used_after_sleep = post[1] - post[0]
		reload_bytes = max(0, used_after - used_after_sleep)
		print(f"[KTSE] 恢复后占用: {(format_gib(total_after)-format_gib(free_after)):.2f} GiB / {format_gib(total_after):.2f} GiB" + (f"  | 重新加载约: {format_gib(reload_bytes)} GiB" if reload_bytes is not None else ""))

	print("[KTSE] 测试完成！ ")
	return {
		"backend": backend,
		"ttft_s": ttft_s,
		"same": same,
		"release_bytes": release_bytes,
		"reload_bytes": reload_bytes,
		"free_before": info0[0] if info0 else None,
		"free_after_sleep": free_after_sleep,
		"free_after_wake": free_after_wake,
		"baseline_preview": baseline_text[:200],
		"text_after_preview": text_after[:200],
	}

if __name__ == "__main__":
	_ = get_device_backend()
	test()