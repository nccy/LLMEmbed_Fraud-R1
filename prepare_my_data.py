import pandas as pd
from datasets import Dataset, DatasetDict

# ==========================================
# 1. 读取原始 CSV 数据 (已经分好了训练集和测试集)
# ==========================================
train_path = "欺诈通话数据集/训练集结果.csv"
test_path = "欺诈通话数据集/测试集结果.csv"

print("正在读取 CSV 文件...")
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# 清理空数据：去掉对话内容为空的行
df_train = df_train.dropna(subset=['specific_dialogue_content'])
df_test = df_test.dropna(subset=['specific_dialogue_content'])

# ==========================================
# 2. 制作【任务A】的数据集 (二分类：诈骗 vs 非诈骗)
# ==========================================
print("\n正在制作任务 A (二分类) 的数据集...")

def process_task_a(df):
    df_a = df[['specific_dialogue_content', 'is_fraud']].copy()
    # 将 TRUE/FALSE 转换为 1 和 0 (兼容字符串和布尔值读取)
    df_a['label'] = df_a['is_fraud'].apply(lambda x: 1 if str(x).strip().upper() == 'TRUE' or x == True else 0)
    df_a = df_a.rename(columns={'specific_dialogue_content': 'text'})
    return df_a[['text', 'label']]

train_a = process_task_a(df_train)
test_a = process_task_a(df_test)

# 直接拼接为 Hugging Face 数据集
dataset_a = DatasetDict({
    'train': Dataset.from_pandas(train_a, preserve_index=False),
    'test': Dataset.from_pandas(test_a, preserve_index=False)
})

# 直接保存在当前目录的 fraud_binary 文件夹下
dataset_a.save_to_disk("./fraud_binary")
print("任务 A 数据集保存成功！路径：./fraud_binary")


# ==========================================
# 3. 制作【任务B】的数据集 (多分类：具体诈骗类型)
# ==========================================
print("\n正在制作任务 B (多分类) 的数据集...")

# 任务B只需要“已经是诈骗”的电话来进行细分类
df_train_b = df_train[df_train['is_fraud'].apply(lambda x: str(x).strip().upper() == 'TRUE' or x == True)].copy()
df_test_b = df_test[df_test['is_fraud'].apply(lambda x: str(x).strip().upper() == 'TRUE' or x == True)].copy()

# 去掉没有填写 fraud_type 的行
df_train_b = df_train_b.dropna(subset=['fraud_type'])
df_test_b = df_test_b.dropna(subset=['fraud_type'])

df_train_b = df_train_b.rename(columns={'specific_dialogue_content': 'text'})
df_test_b = df_test_b.rename(columns={'specific_dialogue_content': 'text'})

# 获取所有可能的诈骗类型，建立统一的映射字典
all_fraud_types = pd.concat([df_train_b['fraud_type'], df_test_b['fraud_type']]).unique().tolist()
type_to_id = {t: i for i, t in enumerate(all_fraud_types)}

print("【重要】任务B的标签映射关系：", type_to_id)

# 映射标签为整数 (0, 1, 2...)
df_train_b['label'] = df_train_b['fraud_type'].map(type_to_id)
df_test_b['label'] = df_test_b['fraud_type'].map(type_to_id)

dataset_b = DatasetDict({
    'train': Dataset.from_pandas(df_train_b[['text', 'label']], preserve_index=False),
    'test': Dataset.from_pandas(df_test_b[['text', 'label']], preserve_index=False)
})

# 直接保存在当前目录的 fraud_multi 文件夹下
dataset_b.save_to_disk("./fraud_multi")
print("任务 B 数据集保存成功！路径：./fraud_multi")
print(f"任务 B 一共有 {len(all_fraud_types)} 个诈骗类别。")