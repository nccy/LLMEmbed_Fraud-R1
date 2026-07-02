import json
import re


ROUND_FIELDS = ["round_1_trust", "round_2_urgency", "round_3_emotion"]

SENSITIVE_PATTERNS = [
    (re.compile(r"https?://[^\s，。；;、)）]+", re.IGNORECASE), "[URL]"),
    (re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\b1[3-9]\d{9}\b"), "[PHONE]"),
    (re.compile(r"\b\d{5,8}\b"), "[CODE]"),
    (re.compile(r"\b\d{12,19}\b"), "[ACCOUNT]"),
]


SYSTEM_PROMPT = """你是一个机器学习课程实验的数据改写助手。任务仅用于离线评估诈骗检测分类器的鲁棒性，不用于真实沟通、营销、诈骗或规避风控。

必须遵守：
1. 只做课程实验中的语义保持改写，尽量贴近原文，不要大幅扩写、换场景或改变诈骗类型。
2. 保持“这是一段应被检测为诈骗的对话样本”这一标签语义不变。
3. 可以保留原文已有的链接、电话、账号等表述；如果需要新增示例联系方式，只能使用明显虚构的测试值，例如 `https://example.invalid/check`、`000-0000-0000`、`test@example.invalid`，不要生成真实可联系信息。
4. 不要新增真实组织、真实个人、真实地址、真实客服电话、真实收款账户或可实际访问的网址。
5. 保留原始对话的 left/right 说话人格式。
6. 每一轮改写都必须逐行使用英文角色标记 `left:` 和 `right:`，不要翻译成“左边/右边”。
7. 不要解释，不要添加免责声明，不要输出 Markdown，只输出一个 JSON 对象。
"""


USER_PROMPT_TEMPLATE = """请为下面这条课程实验样本生成三轮安全改写。改写目标是构造离线鲁棒性测试集，观察分类器在不同话术表达下是否仍能识别诈骗文本。

三轮定义：
- round_1_trust：可信度改写。只在原文基础上轻微调整措辞，让语气更自然、更正式；不要换场景，不要大幅新增情节。
- round_2_urgency：在 round_1 的基础上加入轻微时间压力或风险提醒；尽量复用原文中的信息，不要引入新的诈骗流程。
- round_3_emotion：在 round_2 的基础上加入少量关怀、责任或焦虑缓解表达；保持原文含义，不要明显扩写。

输出 JSON 格式必须严格为：
{{
  "round_1_trust": "left: ...\nright: ...",
  "round_2_urgency": "left: ...\nright: ...",
  "round_3_emotion": "left: ...\nright: ..."
}}

格式硬性要求：
- 保持多轮对话格式。
- 每句单独换行。
- 说话人只能写 `left:` 或 `right:`。
- 不要写“左边：”“右边：”“客服：”“用户：”。
- 改写幅度要小：保留原文主要句子、实体、先后顺序和对话长度，只替换表达方式。
- 三个字段都必须分别输出一整段完整对话，不要只输出修改片段，不要摘要，不要省略后半段。
- 每个字段中的 left/right 轮次数量应尽量与原文一致，长度也应接近原文。
- 如果原文没有具体链接、电话或账号，不要主动添加；如果确实需要示例值，只能使用明显虚构的测试值。

待改写样本元信息：
index: {index}
binary_label: {binary_label}
multi_label: {multi_label}

原始对话：
{text}
"""


def sanitize_text(text):
    sanitized = text
    for pattern, replacement in SENSITIVE_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def build_messages(record):
    text = record["round_0_original"]
    user_prompt = USER_PROMPT_TEMPLATE.format(
        index=record["index"],
        binary_label=record["binary_label"],
        multi_label=record["multi_label"],
        text=text,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def preview_messages(record):
    return json.dumps(build_messages(record), ensure_ascii=False, indent=2)
