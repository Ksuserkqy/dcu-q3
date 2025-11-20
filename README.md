---
prev:
  text: '第二题'
  link: '/dcu/q2'
next:
  text: '总结'
  link: '/dcu/summary'
---

如您将此项目运行在 `海光DCU` 环境中，请勿使用 `requirements.txt` 进行依赖安装，详情请参考[说明文档](https://www.ksuser.cn/dcu/q3.html)。

# 题目3:  `大模型动态权重迁移与推理延迟评测`   <Badge type="warning" text="40%" />
在大模型推理部署场景中，为避免模型 **`权重`长时间静置于 GPU 显存而造成的算力浪费**，因此需设计并实现一套 **`动态权重迁移机制`** 。该机制需具备在`模型空闲时`将 **`权重数据`从 GPU 显存迁移至主机内存**，并在新推理请求到达时**迅速**将权重**恢复**回 **GPU 显存完成推理计算的能力** ；
- 基于 [DeepSeek-R1-Distill-Qwen-7B](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) 模型，所有评测均采用统一的测试输入 **`The future of artificial intelligence is`** ；
- 要求模型连续生成 50 个 token 作为输出。参考开源实现如 [vLLM 的“睡眠模式”机制](https://vllm-ascend-cn.readthedocs.io/zh-cn/latest/user_guide/feature_guide/sleep_mode.html) ；
## 评判标准
- **功能实现**：迁移与恢复过程中是否会推理错误，若能**完整生成**并且输出**内容风格与基线一致**则视为正确 ；
- **性能指标**：核心评测首 `token 延迟`，即从推理请求到达、模型开始恢复，到输出首个 token 的总时间。延迟必须**小于 `5` 秒**，延迟越小越优 ；
- **迁移判定准确性**：通过监控**显存使用量**，验证空闲时显存释放的数据量，恢复后显存是否重新加载模型参数 

## 项目源代码拉取
```bash
$ git clone https://github.com/Ksuserkqy/dcu-q3.git
```

### 目录结构
```bash
.
── q3
   ├─ models            （在执行download.py后才会生成）
   │  └─ deepseek-ai
   │     └─ DeepSeek-R1-Distill-Qwen-7B
   ├─ settings.py       设置（模型名称、token生成相关参数等）
   ├─ tools.py          公告工具函数
   ├─ download.py       下载基准模型/数据集
   ├─ ktse.py           睡眠模式核心引擎(KsuserTransformersSleepEngine)
   ├─ idle.py           定时检测模型是否持续空闲
   ├─ main.py           离线推理/测试TTFT、占用部分
   ├─ web.py            在线服务模块
   ├─ requirements.txt  项目依赖(仅当您在非海光DCU环境部署时才可使用)  # [!code warning] 
   └─ README.md         部分部署的注意事项
```

### 配置文件 `settings.py` 部分参数配置说明
- `ASSERT_TTFT` = False/True # 当 TTFT大于5时是否直接退出进程
- `KEEP_LAYERS` = int # 睡眠时保留在 GPU 上的层数，0表示全部迁移到 CPU

## 实际部署
:::warning 注意
如您在 `海光DCU` 环境上，您需要进行专用的环境检测与依赖安装脚本（详见请至：[第三题环境配置](https://www.ksuser.cn/dcu/env.html#q3-env)）

该项目强烈推荐使用**多卡**，如您坚持使用单卡，请确保您显卡的显存至少有16GB以上（不包含），
```bash
$ curl -fsSL https://static.ksuser.cn/dcu/env-q3.sh | bash -s --
```
接下来的**所有步骤**都需要在脚本创建的 **`dcu-q3`** 虚拟环境中进行！
:::
### 1. 下载基准模型与数据集 {#step-1}
您可借助 `download.py` 一键下载模型 `DeepSeek-R1-Distill-Qwen-7B` 与数据集 `wikitext-103-v1/test-00000-of-00001.parquet`，也可自行下载并存放与项目相关目录
:::code-group
```bash [(dcu-q3)]
$ python download.py
```
```bash [输出日志示例]
(dcu-q3) root@worker-0:/public/home/xdzs2025_sspu_jy/work/q3# python download.py 
Downloading Model from https://www.modelscope.cn to directory: /public/home/xdzs2025_sspu_jy/work/q3/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
2025-10-20 01:57:04,504 - modelscope - INFO - Got 11 files, start to download ...
Downloading [configuration.json]: 100%|█████████████████████████████████████████████████████████████| 73.0/73.0 [00:01<00:00, 62.3B/s]
Downloading [config.json]: 100%|███████████████████████████████████████████████████████████████████████| 680/680 [00:01<00:00, 578B/s]
Downloading [LICENSE]: 100%|███████████████████████████████████████████████████████████████████████| 1.04k/1.04k [00:01<00:00, 899B/s]
Downloading [generation_config.json]: 100%|████████████████████████████████████████████████████████████| 181/181 [00:01<00:00, 150B/s]
Downloading [model.safetensors.index.json]: 100%|████████████████████████████████████████████████| 27.4k/27.4k [00:01<00:00, 23.3kB/s]
Downloading [figures/benchmark.jpg]: 100%|██████████████████████████████████████████████████████████| 759k/759k [00:01<00:00, 633kB/s]
Downloading [tokenizer_config.json]: 100%|███████████████████████████████████████████████████████| 3.00k/3.00k [00:00<00:00, 7.11kB/s]
Downloading [README.md]: 100%|███████████████████████████████████████████████████████████████████| 15.6k/15.6k [00:00<00:00, 26.4kB/s]
Downloading [tokenizer.json]: 100%|██████████████████████████████████████████████████████████████| 6.71M/6.71M [00:01<00:00, 6.22MB/s]
Downloading [model-00001-of-000002.safetensors]: 100%|███████████████████████████████████████████| 8.02G/8.02G [04:57<00:00, 28.9MB/s]
Downloading [model-00002-of-000002.safetensors]: 100%|███████████████████████████████████████████| 6.17G/6.17G [05:36<00:00, 19.7MB/s]
Processing 11 items: 100%|█████████████████████████████████████████████████████████████████████████| 11.0/11.0 [05:36<00:00, 30.6s/it]
2025-10-20 02:02:41,352 - modelscope - INFO - Download model 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B' successfully.1<10:02, 14.3MB/s]
模型已下载至/public/home/xdzs2025_sspu_jy/work/q3/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 19.0M/8.02G [00:01<07:05, 20.2MB/s]
```
:::
运行完毕后您将会在项目根目录中看见 `/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` 模型目录

### 2. 对模型进行睡眠模式测评(TTFT与显存占用) {#step-2}
:::code-group
```bash [(dcu-q3)]
$ python main.py
```
```bash [输出日志示例]
(dcu-q3) root@worker-0:/public/home/xdzs2025_sspu_jy/work/q3# python main.py
[KTSE] 模型将从 ./models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B 加载
/opt/conda/envs/dcu-q3/lib/python3.10/site-packages/accelerate/utils/modeling.py:804: UserWarning: expandable_segments not supported on this platform (Triggered internally at /home/pytorch/c10/hip/HIPAllocatorConfig.h:30.)
  _ = torch.tensor([0], device=i)
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████| 2/2 [00:28<00:00, 14.07s/it]
[KTSE] 当前设备被识别为 cuda
[KTSE] 当前显存占用情况 15.31 GiB / 63.94 GiB
[KTSE] 基线生成中……
The following generation flags are not valid and may be ignored: ['temperature', 'top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
Setting `pad_token_id` to `eos_token_id`:151643 for open-end generation.
/opt/conda/envs/dcu-q3/lib/python3.10/site-packages/transformers/models/qwen2/modeling_qwen2.py:301: UserWarning: Attempting to use hipBLASLt on an unsupported architecture! Overriding blas backend to hipblas (Triggered internally at /home/pytorch/aten/src/ATen/Context.cpp:296.)
  freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
/opt/conda/envs/dcu-q3/lib/python3.10/site-packages/transformers/integrations/sdpa_attention.py:83: UserWarning: Flash attention was not supported for current architecture, attempting to run on torch native impl for backend=math (Triggered internally at /home/pytorch/aten/src/ATen/native/transformers/hip/sdp_utils.cpp:216.)
  attn_output = torch.nn.functional.scaled_dot_product_attention(
/opt/conda/envs/dcu-q3/lib/python3.10/site-packages/transformers/integrations/sdpa_attention.py:83: UserWarning: 1Torch was not compiled with memory efficient attention. (Triggered internally at /home/pytorch/aten/src/ATen/native/transformers/hip/sdp_utils.cpp:663.)
  attn_output = torch.nn.functional.scaled_dot_product_attention(
[KTSE] 基线输出：The future of artificial intelligence is likely to be dominated by the concept of "deep learning." The idea is that by training neural networks on massive amounts of data, we can create systems that perform tasks like image recognition, speech recognition, and natural language processing with high accuracy. However,
[KTSE] 进入睡眠模式 …
[KTSE] 入睡后占用: 1.87 GiB / 63.94 GiB  | 释放约: 14.13 GiB
[阶段] 测量 TTFT 中 ...
/opt/conda/envs/dcu-q3/lib/python3.10/site-packages/accelerate/utils/modeling.py:1598: UserWarning: The following device_map keys do not match any submodules in the model: ['a', 'u', 't', 'o']
  warnings.warn(
/opt/conda/envs/dcu-q3/lib/python3.10/site-packages/torch/nn/modules/module.py:935: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at /home/pytorch/build/aten/src/ATen/core/TensorBody.h:489.)
  param_grad = param.grad
Setting `pad_token_id` to `eos_token_id`:151643 for open-end generation.
[KTSE] * 测量完成！ TTFT ≈ 1.438 s （阈值<5s）
[KTSE] 恢复后完整生成对比
Setting `pad_token_id` to `eos_token_id`:151643 for open-end generation.
[KTSE] 唤醒后输出： The future of artificial intelligence is likely to be dominated by the concept of "deep learning." The idea is that by training neural networks on massive amounts of data, we can create systems that perform tasks like image recognition, speech recognition, and natural language processing with high accuracy. However,
[KTSE] 与基线一致: True
[KTSE] 恢复后占用: 16.23 GiB / 63.94 GiB  | 重新加载约: 14.36 GiB
[KTSE] 测试完成！
```
:::
由此可见 TTFT为 `1.438` < 5s，同时在睡眠过程中也大幅降低了显存占用，完美实现了 **`动态权重迁移机制`**

### 3. 启动在线服务并进行相关实验 {#step-3}
您可在命令行通过 `python` 或 `uvicorn` 命令来启动 `Web` 服务
:::code-group
```bash [python]
$ python web.py
```
```bash [uvicorn]
$ uvicorn web:app --host 0.0.0.0 --port 8000
```
:::

## Web服务 API 说明
:::info 注意
空闲检测仅在Web服务中可用
:::
### 睡眠/唤醒
:::code-group
```bash [睡眠]
$ curl -X POST http://localhost:8000/sleep
```
```bash [唤醒]
$ curl -X POST http://localhost:8000/wake_up
```
```bash [查看当前是否正在睡眠]
$ curl -X GET http://localhost:8000/is_sleeping
```
:::

### 文本生成
对于Web服务的文本生成，您可自由选择是否流式输出（默认流式）
:::code-group
```bash [流式]
$ curl -X POST http://localhost:8000/completions \
	-H "Content-Type: application/json" \
	-d '{"prompt": "The future of artificial intelligence is", "max_tokens": 50, "stream": true}'
```
```bash [非流式]
$ curl -X POST http://localhost:8000/completions \
	-H "Content-Type: application/json" \
	-d '{"prompt": "The future of artificial intelligence is", "max_tokens": 50, "stream": false}'
```
:::

## 睡眠原理详解
### 核心思路
- **空闲检测**：通过时间记录与 `idle.py` 时刻监控推理队列/最近推理时间来判断模型是否进入“空闲”状态。当满足空闲条件时，启动睡眠（检测灵敏度可在 `settings.py` 中进行修改）
- **睡眠（sleep）**：根据配置的 `KEEP_LAYERS` 参数，选择保留在 GPU 上的最后若干层（可保持少量计算能力以降低唤醒成本），其余层的权重采用 PyTorch/Accelerate 的设备移动接口从 `cuda` 移到 `cpu`。迁移过程中模型模块结构与参数名的不变性，以便恢复时能准确映射
- **唤醒（wakeup）**：收到新推理请求时，先快速将被迁移到主机内存的权重写回到 GPU（可并行多线程或分块加载以缩短首 token 延迟），等待关键层参数就绪后触发推理计算。唤醒完成后再次校验显存占用，确保参数完整加载

### 实现要点与安全约束
- **最小保留层**：`KEEP_LAYERS` 控制入睡时保留在 GPU 上的最后 N 层，值为 0 时表示尝试将全部参数迁移至主机内存（更节省显存但唤醒代价更高）。推荐在低显存设备上将该值调高以避免过长的 TTFT
- **阀值保护**：`ASSERT_TTFT` 若为 True 则在测得 TTFT > 5s 时直接退出，默认为 False

### 核心引擎
- **`ktse.KsuserTransformersSleepEngine`**: 睡眠引擎的核心类，负责检测、迁移与恢复
- 主要方法：
  - `sleep()`: 将超过 KEEP_LAYERS 的层权重迁移至主机内存；
  - `wake()`: 将迁移的权重恢复至 GPU；
  - `generate_prepared`: 内容生成准备
  - `generate`: 生成非流式内容
  - `generate_stream`: 生成流式内容
  - `touch_activity`: 模型活动标题（供Web服务调用）
  - `is_sleeping()`: 返回当前引擎是否处于睡眠态

### TTFT测量方法
- TTFT（Time To First Token）= 从收到推理请求开始（包括唤醒/恢复模型的时间）到输出第一个 token 所需的总时间
- 测量步骤：
  1. 确保模型已进入睡眠态（`sleep()` 执行完成且 `is_sleeping()` 为 True）
  2. 在单独计时器中调用 `measure_ttft(prompt)` 或等价流程：记录 t0；触发 `wake_up()`；开始推理并在接收到首个 token 时记录 t1
  3. TTFT = t1 - t0
- 要点：测量时关闭或固定随机性（temperature、top_p 等），并在短时间内多次测量以排除缓存/IO 波动的影响

### 显存占用验证
- 在迁移前后对比显存使用量：本项目通过 `torch.cuda.memory_allocated()` 获取显存占用，当然您也可以使用题目二中的 `rocm-smi` 命令查看

## 调常见问题与排障
- **首 token 时间过长**
  - 通过日志打印分块加载时间，定位是 IO（磁盘->主机内存）还是主机->设备（cpu->cuda）传输耗时，或是模型初始化/编译耗时（例如第一次调用 attention kernel）
- **显存不一致**（恢复后显存没有完全回到迁移前）
  - 检查：确认所有参数已成功加载；检查是否有临时缓存或其余进程占用显存（可以在唤醒后运行 `torch.cuda.empty_cache()` 并再次测量）
- 爆显存：
  - 可以修改 `KEEP_LAYERS` 的大小（调高）
    :::code-group
    ```python:line-numbers=14 {1} [settings.py]
    # Advanced
    TTFT_TOKENS = 1
    ASSERT_TTFT = False # 当 TTFT大于5时是否直接退出进程
    KEEP_LAYERS = 0 # [!code ++] [!code focus] 睡眠时保留在 GPU 上的层数，0表示全部迁移到 CPU
    IDLE_TIMEOUT_SECONDS = 300 # 超过多少秒无活动后触发模型 sleep（默认 300s = 5min）
    IDLE_CHECK_INTERVAL_SECONDS = 5 # 后台检查间隔（秒），建议不要太小以免频繁轮询
    ```
    :::
    :::warning TTFT——显存平衡策略
    随着 `KEEP_LAYERS` 升高，`TTFT` 会随之下降，但 `显存占用` 也会随之升高，因此请平衡好该参数
    :::
  - 或编辑 `ktse.py` 中的第149行，修改 `auto` 为 `balanced` 或 `balanced_low_0`
    :::code-group
    ```python:line-numbers=145 {1} [ktse.py]
    def wake(self) -> None:
		"""将模型权重从 CPU 恢复到原始 device_map"""
		try:
			# [!code focus] 在此处修改device_map参数 
			self.model = dispatch_model(self.model, device_map="auto")  # [!code ++] [!code focus] 在此处修改device_map参数 
			self._sleeping = False
			return
		except Exception as error:
			pass
    ```
    :::

## 许可与引用
- 本项目遵行 **[Apache License 2.0协议](https://www.apache.org/licenses/LICENSE-2.0)**
- 模型遵循其的开源协议与使用条款
- 该项目引用与参考了下列开源项目文档：
  - [vLLM 的“睡眠模式”机制](https://vllm-ascend-cn.readthedocs.io/zh-cn/latest/user_guide/feature_guide/sleep_mode.html)