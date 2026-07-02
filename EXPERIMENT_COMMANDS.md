# LLMEmbed 诈骗检测实验可复现命令

本文档记录当前实验主线的可复现命令。默认在仓库根目录运行：

```bash
cd LLMEmbed-ACL2024-main
```

当前可用运行环境为：

```bash
conda activate llmembed
```

也可以使用非交互形式：

```bash
conda run -n llmembed <command>
```

## 1. 基线训练

二分类诈骗检测：

```bash
python main.py 0 fraud_binary 10 0.1 1024 0.0001 --seed 42
```

多分类诈骗类型识别：

```bash
python main.py 1 fraud_multi 10 0.1 1024 0.0001 --seed 42
```

当前已完成实验使用的 checkpoint：

```text
checkpoints/fraud_binary_20260621_170229.pt
checkpoints/fraud_multi_20260621_170229.pt
```

## 2. 基线评估复算

二分类测试集评估并保存预测：

```bash
python evaluate_checkpoint.py checkpoints/fraud_binary_20260621_170229.pt 0 \
  --split test \
  --prediction_output predictions/fraud_binary_20260621_170229_test.json
```

多分类测试集评估并保存预测：

```bash
python evaluate_checkpoint.py checkpoints/fraud_multi_20260621_170229.pt 1 \
  --split test \
  --prediction_output predictions/fraud_multi_20260621_170229_test.json
```

当前基线结果可通过上述命令在本地复算；对抗评估汇总见：

```text
adversarial_data/evaluation/adv_evaluation_summary.md
```

## 3. 阶段 2：导出改写攻击源数据

从课程数据集中导出 `fraud_binary/test` 中 `label=1` 的诈骗样本，并与 `fraud_multi/test` 的诈骗类型标签对齐：

```bash
python adversarial_rewrite/export_attack_source.py
```

默认输出：

```text
adversarial_data/source/fraud_test_attack_source.json
adversarial_data/source/fraud_test_attack_source.summary.json
```

预期样本数：

```text
1387
```

## 4. 阶段 3：生成改写样本

当前提示词策略：尽量在原文基础上做小幅语义保持改写，不强制把链接、电话、账号替换为占位符；如果模型需要新增示例联系方式，只允许使用明显虚构的测试值，避免真实可联系信息。

如果使用本地 Ollama，先启动 GPU 服务。当前 GPU 使用约束为：只使用 `0,1,2` 号 GPU，不使用后续 GPU。下面示例在 GPU 0 上启动一个服务，并显式绕过 localhost 代理：

```bash
setsid env \
  CUDA_VISIBLE_DEVICES=0 \
  GGML_VK_VISIBLE_DEVICES=0 \
  OLLAMA_VULKAN=false \
  OLLAMA_CONTEXT_LENGTH=4096 \
  OLLAMA_FLASH_ATTENTION=true \
  NO_PROXY=localhost,127.0.0.1 \
  no_proxy=localhost,127.0.0.1 \
  OLLAMA_HOST=http://127.0.0.1:11434 \
  ollama serve > logs/ollama_gpu0_11434.log 2>&1 < /dev/null &
```

检查服务和 GPU 使用：

```bash
NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 \
  curl -sS http://127.0.0.1:11434/api/version

ollama ps
nvidia-smi
```

先检查提示词，不调用模型：

```bash
python adversarial_rewrite/rewrite_with_llm.py --max_items 10 --dry_run
```

使用本地 Ollama OpenAI-compatible 接口生成 10 条烟测样本：

```bash
python adversarial_rewrite/rewrite_with_llm.py \
  --provider ollama \
  --model qwen3.5:9b \
  --max_items 10 \
  --json_mode \
  --timeout 300 \
  --retries 1 \
  --output adversarial_data/rewrites/smoke_qwen3.5-9b_fraud_test_rewrites.json
```

如果本地 `/v1/chat/completions` 兼容接口不可用，可改用 Ollama 原生接口：

```bash
python adversarial_rewrite/rewrite_with_llm.py \
  --provider ollama-native \
  --model llama3.1:8b \
  --max_items 10 \
  --json_mode \
  --timeout 300 \
  --retries 1 \
  --output adversarial_data/rewrites/smoke_llama3.1-8b_fraud_test_rewrites.json
```

使用在线 OpenAI-compatible API：

```bash
export REWRITE_BASE_URL="https://your-api-host/v1"
export REWRITE_API_KEY="your-api-key"

python adversarial_rewrite/rewrite_with_llm.py \
  --provider openai-compatible \
  --model your-model-name \
  --max_items 10 \
  --json_mode \
  --output adversarial_data/rewrites/smoke_online_fraud_test_rewrites.json
```

完整 1387 条样本生成时建议开启断点续跑：

```bash
python adversarial_rewrite/rewrite_with_llm.py \
  --provider ollama-native \
  --model llama3.1:8b \
  --json_mode \
  --resume \
  --output adversarial_data/rewrites/llama3.1-8b_fraud_test_rewrites.json
```

三路本地模型并行生成完整 1387 条样本时，使用独立 Ollama 端口和独立输出文件，避免互相抢占同一服务。端口映射必须限制在 GPU 0/1/2：

```text
127.0.0.1:11434 -> llama3.1:8b    -> GPU 0
127.0.0.1:11435 -> qwen3.5:9b     -> GPU 1
127.0.0.1:11436 -> deepseek-r1:8b -> GPU 2
```

正式并行启动命令：

```bash
setsid env OLLAMA_URL=http://127.0.0.1:11434 \
  PYTHONUNBUFFERED=1 \
  NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 \
  conda run -n llmembed python adversarial_rewrite/rewrite_with_llm.py \
    --provider ollama-native \
    --model llama3.1:8b \
    --json_mode \
    --temperature 0 \
    --timeout 300 \
    --retries 2 \
    --retry_sleep 10 \
    --save_every 1 \
    --resume \
    --output adversarial_data/rewrites/llama3.1-8b_fraud_test_rewrites.json \
  > logs/rewrite_llama3.1-8b.log 2>&1 < /dev/null &

setsid env OLLAMA_URL=http://127.0.0.1:11435 \
  PYTHONUNBUFFERED=1 \
  NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 \
  conda run -n llmembed python adversarial_rewrite/rewrite_with_llm.py \
    --provider ollama-native \
    --model qwen3.5:9b \
    --json_mode \
    --temperature 0 \
    --timeout 300 \
    --retries 2 \
    --retry_sleep 10 \
    --save_every 1 \
    --resume \
    --output adversarial_data/rewrites/qwen3.5-9b_fraud_test_rewrites.json \
  > logs/rewrite_qwen3.5-9b.log 2>&1 < /dev/null &

setsid env OLLAMA_URL=http://127.0.0.1:11436 \
  PYTHONUNBUFFERED=1 \
  NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 \
  conda run -n llmembed python adversarial_rewrite/rewrite_with_llm.py \
    --provider ollama-native \
    --model deepseek-r1:8b \
    --temperature 0 \
    --max_tokens 16384 \
    --num_ctx 32768 \
    --timeout 600 \
    --retries 2 \
    --retry_sleep 10 \
    --save_every 1 \
    --resume \
    --output adversarial_data/rewrites/deepseek-r1-8b_fraud_test_rewrites.json \
  > logs/rewrite_deepseek-r1-8b.log 2>&1 < /dev/null &
```

进度检查：

```bash
ps -ef | grep rewrite_with_llm | grep -v grep
tail -f logs/rewrite_llama3.1-8b.log
tail -f logs/rewrite_qwen3.5-9b.log
tail -f logs/rewrite_deepseek-r1-8b.log

python - <<'PY'
import json
from pathlib import Path

for path in [
    "adversarial_data/rewrites/llama3.1-8b_fraud_test_rewrites.json",
    "adversarial_data/rewrites/qwen3.5-9b_fraud_test_rewrites.json",
    "adversarial_data/rewrites/deepseek-r1-8b_fraud_test_rewrites.json",
]:
    p = Path(path)
    if not p.exists():
        print(path, "missing")
        continue
    data = json.load(p.open(encoding="utf-8"))
    valid = sum(1 for row in data if row.get("is_valid"))
    print(path, f"{len(data)}/1387 records", f"valid={valid}")
PY
```

如果启动时没有设置 `PYTHONUNBUFFERED=1`，日志可能到任务结束才集中刷新；这种情况下以上 Python 片段检查 JSON 输出文件是更可靠的实时进度来源。

`deepseek-r1:8b` 使用与其他模型完全相同的提示词，但不要启用 `--json_mode`。该模型在 Ollama native `format=json` 和 4096 上下文下容易进入长 thinking、输出空 `content` 或异常问号；正式生成使用 `--max_tokens 16384 --num_ctx 32768`，让 thinking 和最终 JSON 都落在同一上下文窗口内。低 token / json_mode 配置产生的失败文件已备份到：

```text
adversarial_data/rewrites/diagnostics/deepseek-r1-8b_fraud_test_rewrites.failed_max2048_20260626_1836.json
adversarial_data/rewrites/diagnostics/deepseek-r1-8b_fraud_test_rewrites.bad_jsonmode_ctx4096_20260626_2000.json
```

注意：本机 shell 中设置了 `http_proxy/https_proxy`。如果没有 `NO_PROXY/no_proxy=localhost,127.0.0.1`，访问本地 Ollama 会被代理转发并返回 `502 Bad Gateway`。`rewrite_with_llm.py` 已对本地 Ollama provider 禁用 requests 的环境代理，但手工 `curl` 时仍建议显式设置 `NO_PROXY`。

2026-06-26 修复后，`llama3.1:8b` 已通过 GPU smoke test：

```text
ollama ps: 100% GPU, context 4096
10 条 smoke: request/parse failures = 0, valid = 9/10
```

历史失败诊断输出保留在：

```text
adversarial_data/rewrites/diagnostics/
```

## 5. 当前改写数据状态

截至 2026-06-29，四个正式改写文件均已补齐到 100% 可用。判定口径为 `is_valid == true`，并用 `retry_invalid_rewrites.py --dry_run` 复核剩余待重试条数为 0：

```text
qwen3.5:9b     total=1387 valid=1387 invalid=0 usable=100.00%
llama3.1:8b   total=1387 valid=1387 invalid=0 usable=100.00%
kimi-k2.5     total=1387 valid=1387 invalid=0 usable=100.00%
glm-5.2       total=1387 valid=1387 invalid=0 usable=100.00%
```

正式进入后续 embedding 重建和鲁棒性评估的改写文件：

```text
adversarial_data/rewrites/qwen3.5-9b_fraud_test_rewrites.json
adversarial_data/rewrites/llama3.1-8b_fraud_test_rewrites.json
adversarial_data/rewrites/kimi-k2.5_fraud_test_rewrites.json
adversarial_data/rewrites/glm-5.2_fraud_test_rewrites.json
```

验证摘要：

```text
adversarial_data/validated/rewrite_validation_summary.md
adversarial_data/validated/rewrite_validation_summary_online.md
```

复核命令：

```bash
conda run -n llmembed python adversarial_rewrite/retry_invalid_rewrites.py \
  --dry_run \
  --models qwen3.5:9b llama3.1:8b kimi-k2.5 glm-5.2
```

预期输出均为：

```text
would retry 0 records
```

### Kimi 和 GLM 质量判断

`kimi-k2.5` 和 `glm-5.2` 的历史瓶颈曾经是在线 API `403 Forbidden`，不是提示词质量。2026-06-29 已补跑完成，两个文件均为 `1387/1387` 有效。已通过样本的结构特征如下：

```text
kimi-k2.5: round_1/2/3 平均字符数 423.8 / 469.0 / 532.0，平均对话行数 8.3
glm-5.2:   round_1/2/3 平均字符数 441.1 / 470.7 / 519.8，平均对话行数 8.2
```

## 6. 阶段 5：使用 100% 通过率改写数据重建 embedding

阶段 5 已覆盖四个 100% 通过率改写模型：`qwen3.5:9b`、`llama3.1:8b`、`kimi-k2.5`、`glm-5.2`。

阶段 5 新增脚本：

```text
adversarial_rewrite/build_adv_dataset.py
```

默认只重建三轮改写文本，不包含 `round_0_original`：

```text
round_1_trust
round_2_urgency
round_3_emotion
```

当前进入重建的 100% 通过率改写文件：

```text
adversarial_data/rewrites/qwen3.5-9b_fraud_test_rewrites.json
adversarial_data/rewrites/llama3.1-8b_fraud_test_rewrites.json
adversarial_data/rewrites/kimi-k2.5_fraud_test_rewrites.json
adversarial_data/rewrites/glm-5.2_fraud_test_rewrites.json
```

每个改写模型、每个轮次会生成 6 个任务目录：三路 encoder 各写入 `fraud_binary_adv_*` 和 `fraud_multi_adv_*`。二分类任务保留完整 `fraud_binary/test`，`label=0` 非诈骗样本原样保留，`label=1` 诈骗样本替换为对应改写文本；多分类任务仍为 1387 条诈骗样本。示例：

```text
bert_embedding/fraud_binary_adv_qwen3.5-9b_r1_trust/dataset_tensor/test_sents.pt
bert_embedding/fraud_binary_adv_qwen3.5-9b_r1_trust/dataset_tensor/test_labels.pt
bert_embedding/fraud_multi_adv_qwen3.5-9b_r1_trust/dataset_tensor/test_sents.pt
bert_embedding/fraud_multi_adv_qwen3.5-9b_r1_trust/dataset_tensor/test_labels.pt
```

二分类标签使用原始 `fraud_binary/test` 标签，分布保持 `label=0`: 1290、`label=1`: 1387；多分类标签使用 `multi_label`。同一轮先抽取完整二分类文本 embedding，再按 `binary_index` 取出诈骗子集保存到多分类任务目录，避免重复计算。

### 6.1 重建前检查

确认四个模型均为 100% 可用：

```bash
conda run -n llmembed python adversarial_rewrite/retry_invalid_rewrites.py \
  --dry_run \
  --models qwen3.5:9b llama3.1:8b kimi-k2.5 glm-5.2
```

确认 CUDA 可用：

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader

conda run -n llmembed python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

2026-06-29 已确认 `conda run -n llmembed` 下 CUDA 可用。后续脚本已经加入 GPU 白名单，`cuda_no` 只能为 `0,1,2`。

### 6.2 按 encoder 分步重建

建议按 encoder 分开跑，便于失败后断点重跑。下面命令会同时写出二分类和多分类任务目录。

以下命令展示按 encoder 分步重建的形式。`qwen3.5:9b`：

```bash
conda run -n llmembed python adversarial_rewrite/build_adv_dataset.py \
  --rewrite_json adversarial_data/rewrites/qwen3.5-9b_fraud_test_rewrites.json \
  --rewrite_model_slug qwen3.5-9b \
  --encoder bert \
  --cuda_no 0

conda run -n llmembed python adversarial_rewrite/build_adv_dataset.py \
  --rewrite_json adversarial_data/rewrites/qwen3.5-9b_fraud_test_rewrites.json \
  --rewrite_model_slug qwen3.5-9b \
  --encoder roberta \
  --cuda_no 1

conda run -n llmembed python adversarial_rewrite/build_adv_dataset.py \
  --rewrite_json adversarial_data/rewrites/qwen3.5-9b_fraud_test_rewrites.json \
  --rewrite_model_slug qwen3.5-9b \
  --encoder llama2 \
  --cuda_no 2 \
  --rounds round_1_trust

conda run -n llmembed python adversarial_rewrite/build_adv_dataset.py \
  --rewrite_json adversarial_data/rewrites/qwen3.5-9b_fraud_test_rewrites.json \
  --rewrite_model_slug qwen3.5-9b \
  --encoder llama2 \
  --cuda_no 2 \
  --rounds round_2_urgency

conda run -n llmembed python adversarial_rewrite/build_adv_dataset.py \
  --rewrite_json adversarial_data/rewrites/qwen3.5-9b_fraud_test_rewrites.json \
  --rewrite_model_slug qwen3.5-9b \
  --encoder llama2 \
  --cuda_no 2 \
  --rounds round_3_emotion
```

`llama3.1:8b`：

```bash
conda run -n llmembed python adversarial_rewrite/build_adv_dataset.py \
  --rewrite_json adversarial_data/rewrites/llama3.1-8b_fraud_test_rewrites.json \
  --rewrite_model_slug llama3.1-8b \
  --encoder bert \
  --cuda_no 0

conda run -n llmembed python adversarial_rewrite/build_adv_dataset.py \
  --rewrite_json adversarial_data/rewrites/llama3.1-8b_fraud_test_rewrites.json \
  --rewrite_model_slug llama3.1-8b \
  --encoder roberta \
  --cuda_no 1

conda run -n llmembed python adversarial_rewrite/build_adv_dataset.py \
  --rewrite_json adversarial_data/rewrites/llama3.1-8b_fraud_test_rewrites.json \
  --rewrite_model_slug llama3.1-8b \
  --encoder llama2 \
  --cuda_no 2 \
  --rounds round_1_trust

conda run -n llmembed python adversarial_rewrite/build_adv_dataset.py \
  --rewrite_json adversarial_data/rewrites/llama3.1-8b_fraud_test_rewrites.json \
  --rewrite_model_slug llama3.1-8b \
  --encoder llama2 \
  --cuda_no 2 \
  --rounds round_2_urgency

conda run -n llmembed python adversarial_rewrite/build_adv_dataset.py \
  --rewrite_json adversarial_data/rewrites/llama3.1-8b_fraud_test_rewrites.json \
  --rewrite_model_slug llama3.1-8b \
  --encoder llama2 \
  --cuda_no 2 \
  --rounds round_3_emotion
```

`kimi-k2.5` 和 `glm-5.2` 的重建命令同样使用 `build_adv_dataset.py`。2026-06-29 补建时实际执行过的缺口命令如下：

```bash
conda run -n llmembed python adversarial_rewrite/build_adv_dataset.py \
  --rewrite_json adversarial_data/rewrites/kimi-k2.5_fraud_test_rewrites.json \
  --rewrite_model_slug kimi-k2.5 \
  --encoder llama2 \
  --cuda_no 2 \
  --rounds round_2_urgency round_3_emotion

conda run -n llmembed python adversarial_rewrite/build_adv_dataset.py \
  --rewrite_json adversarial_data/rewrites/glm-5.2_fraud_test_rewrites.json \
  --rewrite_model_slug glm-5.2 \
  --encoder bert \
  --cuda_no 0

conda run -n llmembed python adversarial_rewrite/build_adv_dataset.py \
  --rewrite_json adversarial_data/rewrites/glm-5.2_fraud_test_rewrites.json \
  --rewrite_model_slug glm-5.2 \
  --encoder roberta \
  --cuda_no 1

conda run -n llmembed python adversarial_rewrite/build_adv_dataset.py \
  --rewrite_json adversarial_data/rewrites/glm-5.2_fraud_test_rewrites.json \
  --rewrite_model_slug glm-5.2 \
  --encoder llama2 \
  --cuda_no 2
```

小样本烟测只抽取前 3 条：

```bash
conda run -n llmembed python adversarial_rewrite/build_adv_dataset.py \
  --rewrite_json adversarial_data/rewrites/qwen3.5-9b_fraud_test_rewrites.json \
  --rewrite_model_slug qwen3.5-9b_smoke \
  --encoder bert \
  --device cpu \
  --max_items 3 \
  --binary_positive_only \
  --output_root /tmp/llmembed_adv_smoke
```

2026-06-28 已完成上述 CPU smoke test，输出 shape 和标签如下：

```text
fraud_binary_adv_qwen3.5-9b_smoke_r1_trust (3, 1024) [1, 1, 1]
fraud_binary_adv_qwen3.5-9b_smoke_r2_urgency (3, 1024) [1, 1, 1]
fraud_binary_adv_qwen3.5-9b_smoke_r3_emotion (3, 1024) [1, 1, 1]
fraud_multi_adv_qwen3.5-9b_smoke_r1_trust  (3, 1024) [0, 4, 0]
fraud_multi_adv_qwen3.5-9b_smoke_r2_urgency (3, 1024) [0, 4, 0]
fraud_multi_adv_qwen3.5-9b_smoke_r3_emotion (3, 1024) [0, 4, 0]
```

### 6.3 重建后校验

校验所有阶段 5 产物样本数、标签数和三路 shape：

```bash
conda run -n llmembed python adversarial_rewrite/verify_adv_tensors.py \
  --models qwen3.5-9b llama3.1-8b kimi-k2.5 glm-5.2
```

2026-06-29 已完成上述校验，四个模型、三轮改写、二分类/多分类任务均通过；结果写入：

```text
adversarial_data/validated/adv_tensor_summary.json
```

当前阶段完成标准：

```text
- 四个正式改写 JSON 均为 1387/1387 可用。
- 对四个模型的三轮改写完成三路 embedding 重建。
- 每个 adv 任务目录均包含 test_sents.pt、test_labels.pt、test_metadata.json。
- 二分类 adv 任务三路 encoder 的样本数均为 2677，标签分布为 {0:1290, 1:1387}。
- 多分类 adv 任务三路 encoder 的样本数均为 1387，并与类型标签数一致。
```

## 7. 阶段 6：对抗鲁棒性评估汇总

状态：已完成。完成时间：2026-06-29。

阶段 6 脚本：

```text
adversarial_rewrite/evaluate_adv.py
```

2026-06-29 复用已训练 checkpoint，对四个改写模型的三轮改写数据完成二分类和多分类评估：

```bash
conda run -n llmembed python adversarial_rewrite/evaluate_adv.py \
  --models qwen3.5-9b llama3.1-8b kimi-k2.5 glm-5.2 \
  --cuda_no 0 \
  --save_predictions
```

输出文件：

```text
adversarial_data/evaluation/adv_evaluation_summary.json
adversarial_data/evaluation/adv_evaluation_summary.md
adversarial_data/evaluation/predictions/
```

### 7.1 二分类结果

二分类基线在 `fraud_binary/test` 上为 `Accuracy=0.9739`、`Recall=1.0000`、`FNR=0.0000`。四个改写模型的三轮攻击总体没有明显削弱二分类诈骗检出能力：

```text
qwen3.5-9b:   ASR 0/1387, 1/1387, 1/1387
llama3.1-8b: ASR 2/1387, 0/1387, 0/1387
kimi-k2.5:   ASR 0/1387, 0/1387, 0/1387
glm-5.2:     ASR 0/1387, 0/1387, 0/1387
```

结论：对当前 LLMEmbed 二分类诈骗检测 checkpoint，三轮语义保持改写的攻击成功率最高只有 `0.0014`，整体鲁棒性很强。

### 7.2 多分类结果

多分类基线在 `fraud_multi/test` 上为 `Accuracy=0.6864`、`macro-F1=0.5257`。四个改写模型的三轮结果如下：

```text
qwen3.5-9b   r1/r2/r3 accuracy=0.6503/0.6518/0.6583 macro-F1=0.4654/0.4626/0.4801
llama3.1-8b r1/r2/r3 accuracy=0.6777/0.6748/0.6756 macro-F1=0.5121/0.5132/0.5173
kimi-k2.5   r1/r2/r3 accuracy=0.6698/0.6727/0.6820 macro-F1=0.4941/0.5026/0.5239
glm-5.2     r1/r2/r3 accuracy=0.6727/0.6698/0.6741 macro-F1=0.4998/0.4933/0.5129
```

结论：多分类类型识别比二分类更敏感。`qwen3.5-9b` 改写造成的下降最大，最低 macro-F1 为 `0.4626`；`kimi-k2.5` 的第三轮最接近基线，`Accuracy=0.6820`、`macro-F1=0.5239`。

阶段 6 完成标准：

```text
- 四个正式改写模型均完成二分类和多分类鲁棒性评估。
- 每个模型均覆盖 r1_trust、r2_urgency、r3_emotion 三轮。
- 评估汇总写入 adv_evaluation_summary.json/md。
- 预测明细写入 adversarial_data/evaluation/predictions/。
```

## 8. 阶段 7：改写内容质量复核

状态：已完成。完成时间：2026-06-29。

阶段 7 脚本：

```text
adversarial_rewrite/analyze_rewrite_quality.py
```

该阶段在 `is_valid == true` 的格式校验基础上，进一步检查改写内容是否适合解释对抗攻击结果。主要检查项包括：

```text
- 改写长度与原文长度比例。
- left/right 对话行数是否明显减少。
- 是否保留中文内容。
- 是否存在 near-copy、过度扩写或非角色行。
- 是否保留诈骗相关核心线索，如链接、APP、转账、退款、贷款、投资、收益、威胁、紧急性等。
```

复核命令：

```bash
python adversarial_rewrite/analyze_rewrite_quality.py \
  --rewrite_json \
    adversarial_data/rewrites/qwen3.5-9b_fraud_test_rewrites.json \
    adversarial_data/rewrites/llama3.1-8b_fraud_test_rewrites.json \
    adversarial_data/rewrites/kimi-k2.5_fraud_test_rewrites.json \
    adversarial_data/rewrites/glm-5.2_fraud_test_rewrites.json \
  --max_examples 8
```

输出文件：

```text
adversarial_data/validated/rewrite_quality_analysis.json
adversarial_data/validated/rewrite_quality_analysis.md
```

### 8.1 源样本级可用性

一条源样本包含三轮改写。`Pass all rounds` 表示三轮均未触发质量风险；`Needs review any round` 表示至少一轮存在截断、核心线索丢失或格式异常等问题，需要人工复核。

```text
kimi-k2.5     pass_all_rounds=1313/1387 (94.7%) needs_review_any_round=42/1387  (3.0%)
glm-5.2       pass_all_rounds=1237/1387 (89.2%) needs_review_any_round=23/1387  (1.7%)
qwen3.5-9b    pass_all_rounds=912/1387  (65.8%) needs_review_any_round=186/1387 (13.4%)
llama3.1-8b   pass_all_rounds=499/1387  (36.0%) needs_review_any_round=188/1387 (13.6%)
```

### 8.2 三轮改写项级质量

每个模型共有 `1387 * 3 = 4161` 个改写项：

```text
kimi-k2.5     pass=4014/4161 (96.5%) warning=61  needs_review=86
glm-5.2       pass=3914/4161 (94.1%) warning=191 needs_review=56
qwen3.5-9b    pass=3011/4161 (72.4%) warning=715 needs_review=435
llama3.1-8b   pass=2770/4161 (66.6%) warning=932 needs_review=459
```

### 8.3 质量结论

`kimi-k2.5` 和 `glm-5.2` 的内容质量明显更稳，适合作为高质量改写主结果。`qwen3.5-9b` 存在较多非角色行、英文输出、关键词保留不足和核心线索丢失；`llama3.1-8b` 近复制较多，同时有部分截断和对话轮次下降问题。

`multi_label=3` 是最容易出问题的类型，通常涉及威胁、紧急求助、伤害或转账等强语义线索。此类样本一旦弱化核心线索，就可能从诈骗文本变成普通求助文本。

阶段 7 完成标准：

```text
- 四个正式改写模型均完成内容质量复核。
- 输出轮次级、源样本级、类型级质量指标。
- 标记 needs_review 样本，供解释 attack success 时剔除或单独说明。
- 质量报告写入 rewrite_quality_analysis.json/md。
```
