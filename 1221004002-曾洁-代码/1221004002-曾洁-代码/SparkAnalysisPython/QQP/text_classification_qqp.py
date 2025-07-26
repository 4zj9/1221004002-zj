from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, concat_ws, lit, when
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import time
import os
import gensim
from gensim.models import FastText
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score

# 解析命令行参数
parser = argparse.ArgumentParser(description='Spark文本分类实现 - QQP数据集')
parser.add_argument('--data_file', type=str, default='hdfs://hadoop1:9000/user/spark/glue/QQP/dev.tsv', help='训练数据文件路径')
parser.add_argument('--test_file', type=str, default='hdfs://hadoop1:9000/user/spark/glue/QQP/dev.tsv', help='测试数据文件路径')
parser.add_argument('--epochs', type=int, default=3, help='TextCNN训练轮数')
parser.add_argument('--batch_size', type=int, default=64, help='批处理大小')
parser.add_argument('--fasttext_epochs', type=int, default=3, help='FastText训练轮数')
parser.add_argument('--output_fig', type=str, default='qqp_model_comparison.png', help='输出图表文件名')
args = parser.parse_args()

# 全局变量
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化 Spark 会话
spark = SparkSession.builder \
    .appName("QQP Text Classification") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
    .getOrCreate()


# 加载QQP数据集
def load_qqp_data(file_path, is_test=False):
    print(f"Loading dataset from {file_path}...")
    try:
        # 读取TSV文件
        data = spark.read.csv(file_path, header=True, sep="\t")

        # 如果不是测试集且有is_duplicate列，处理标签
        if not is_test and "is_duplicate" in data.columns:
            # 将标签转换为数值
            data = data.withColumn("label", col("is_duplicate").cast("double"))

            # 过滤掉NULL和NaN值
            data = data.filter(col("label").isNotNull())

            # 确保所有标签都是有效的数值
            data = data.filter((col("label") == 0.0) | (col("label") == 1.0))

        elif is_test:
            # 测试集没有标签，创建虚拟标签
            data = data.withColumn("label", lit(0.0))

        # 合并两个问题为一个文本字段
        data = data.withColumn("text", concat_ws(" [SEP] ", col("question1"), col("question2")))

        # 选择需要的列并确保没有NULL值
        data = data.select("text", "label").filter(col("text").isNotNull())

        # 检查数据集大小
        count = data.count()
        if count == 0:
            raise ValueError("处理后的数据集为空，请检查输入文件和过滤条件")

        return data
    except Exception as e:
        print(f"加载数据时出错: {e}")
        print("创建示例数据集...")

        # 创建一个简单的示例数据集
        data = [("How do I improve my English? [SEP] What are some ways to improve my English?", 1.0),
                ("What is the capital of France? [SEP] What is the population of Germany?", 0.0),
                ("How to lose weight fast? [SEP] What are some effective ways to lose weight quickly?", 1.0)]
        return spark.createDataFrame(data, ["text", "label"])


# 创建评估函数
def evaluate_model(predictions, model_name):
    # 二分类评估
    binary_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
    auc = binary_evaluator.evaluate(predictions)

    # 多分类评估
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    accuracy = evaluator.evaluate(predictions)

    evaluator.setMetricName("weightedPrecision")
    precision = evaluator.evaluate(predictions)

    evaluator.setMetricName("weightedRecall")
    recall = evaluator.evaluate(predictions)

    evaluator.setMetricName("f1")
    f1 = evaluator.evaluate(predictions)

    print(
        f"{model_name} - AUC: {auc:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    return accuracy, precision, recall, f1




# 实现FastText模型
def train_fasttext(train_data, test_data, epochs=3):
    print("\n--- Training FastText Model ---")
    start_time = time.time()

    # 准备数据 - 不使用pyarrow
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
    train_pandas = train_data.toPandas()
    test_pandas = test_data.toPandas()

    # 确保没有NaN值
    train_pandas = train_pandas.dropna()
    test_pandas = test_pandas.dropna()

    # 确保标签是数值类型
    train_pandas["label"] = train_pandas["label"].astype(float)
    test_pandas["label"] = test_pandas["label"].astype(float)

    # 预处理文本
    def preprocess_text(text):
        if isinstance(text, str):
            return text.lower().split()
        return []

    # 创建FastText模型
    train_sentences = [preprocess_text(text) for text in train_pandas["text"]]
    fasttext_model = FastText(sentences=train_sentences, vector_size=100, window=5,
                              min_count=1, workers=4, sg=1, epochs=epochs)

    # 创建文本向量化函数
    def text_to_vector(text, model):
        words = preprocess_text(text)
        word_vectors = [model.wv[word] for word in words if word in model.wv]
        if len(word_vectors) == 0:
            return np.zeros(model.vector_size)
        return np.mean(word_vectors, axis=0)

    # 向量化训练和测试数据
    X_train = np.array([text_to_vector(text, fasttext_model) for text in train_pandas["text"]])
    y_train = train_pandas["label"].values
    X_test = np.array([text_to_vector(text, fasttext_model) for text in test_pandas["text"]])
    y_test = test_pandas["label"].values

    # 使用逻辑回归分类器
    from sklearn.linear_model import LogisticRegression as SklearnLR
    fasttext_clf = SklearnLR(max_iter=100, n_jobs=-1)
    fasttext_clf.fit(X_train, y_train)

    # 预测
    ft_predictions = fasttext_clf.predict(X_test)
    ft_accuracy = np.mean(ft_predictions == y_test)
    ft_precision = precision_score(y_test, ft_predictions, average='weighted')
    ft_recall = recall_score(y_test, ft_predictions, average='weighted')
    ft_f1 = f1_score(y_test, ft_predictions, average='weighted')

    training_time = time.time() - start_time
    print(
        f"FastText - Accuracy: {ft_accuracy:.4f}, Precision: {ft_precision:.4f}, Recall: {ft_recall:.4f}, F1: {ft_f1:.4f}")

    return (ft_accuracy, ft_precision, ft_recall, ft_f1), training_time


# 文本转换为索引
def text_to_indices(text, word_to_idx, max_len=100):
    if not isinstance(text, str):
        return [0] * max_len
    words = text.lower().split()
    indices = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in words[:max_len]]
    padding = [0] * (max_len - len(indices))
    return indices + padding


# 准备数据集
class TextDataset(Dataset):
    def __init__(self, texts, labels, word_to_idx, max_len=100):
        self.texts = [text_to_indices(text, word_to_idx, max_len) for text in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)




# 训练函数
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for batch in iterator:
        optimizer.zero_grad()
        texts, labels = batch
        texts, labels = texts.to(device), labels.to(device)

        predictions = model(texts)
        loss = criterion(predictions, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        _, predicted = torch.max(predictions.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return epoch_loss / len(iterator), correct / total


# 评估函数
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in iterator:
            texts, labels = batch
            texts, labels = texts.to(device), labels.to(device)

            predictions = model(texts)
            loss = criterion(predictions, labels)

            epoch_loss += loss.item()

            _, predicted = torch.max(predictions.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return epoch_loss / len(iterator), accuracy, precision, recall, f1



# 主函数
def main():
    # 加载数据
    train_data = load_qqp_data(args.data_file)
    test_data = load_qqp_data(args.test_file)

    # 显示数据集信息
    print("Training dataset size:", train_data.count())
    print("Test dataset size:", test_data.count())

    # 数据验证 - 检查是否有null值
    null_count_train = train_data.filter(col("label").isNull()).count()
    null_count_test = test_data.filter(col("label").isNull()).count()

    if null_count_train > 0 or null_count_test > 0:
        print(f"警告: 训练集中有 {null_count_train} 行空标签，测试集中有 {null_count_test} 行空标签")
        print("正在过滤空标签...")
        train_data = train_data.filter(col("label").isNotNull())
        test_data = test_data.filter(col("label").isNotNull())
        print("过滤后 - 训练集大小:", train_data.count())
        print("过滤后 - 测试集大小:", test_data.count())

    # 检查标签分布
    label_counts = train_data.groupBy("label").count().collect()
    print("标签分布:")
    for row in label_counts:
        print(f"  标签 {row['label']}: {row['count']} 行")

    train_data.select("text", "label").show(5, truncate=False)

    # 存储结果用于绘图
    results = {
        "Model": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1": [],
        "Training Time": []
    }



    # 2. FastText
    ft_metrics, ft_time = train_fasttext(train_data, test_data, epochs=args.fasttext_epochs)
    results["Model"].append("FastText")
    results["Accuracy"].append(ft_metrics[0])
    results["Precision"].append(ft_metrics[1])
    results["Recall"].append(ft_metrics[2])
    results["F1"].append(ft_metrics[3])
    results["Training Time"].append(ft_time)



    # 绘制结果比较图
    plt.figure(figsize=(14, 10))

    # 准确率对比
    plt.subplot(2, 2, 1)
    plt.bar(results["Model"], results["Accuracy"], color=['blue', 'green', 'red'])
    plt.title('Accuracy Comparison')
    plt.ylim(0, 1)
    for i, v in enumerate(results["Accuracy"]):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')

    # F1分数对比
    plt.subplot(2, 2, 2)
    plt.bar(results["Model"], results["F1"], color=['blue', 'green', 'red'])
    plt.title('F1 Score Comparison')
    plt.ylim(0, 1)
    for i, v in enumerate(results["F1"]):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')

    # 训练时间对比
    plt.subplot(2, 2, 3)
    plt.bar(results["Model"], results["Training Time"], color=['blue', 'green', 'red'])
    plt.title('Training Time Comparison (seconds)')
    for i, v in enumerate(results["Training Time"]):
        plt.text(i, v + 0.5, f'{v:.2f}s', ha='center')

    # 精确率和召回率对比
    plt.subplot(2, 2, 4)
    x = np.arange(len(results["Model"]))
    width = 0.35
    plt.bar(x - width / 2, results["Precision"], width, label='Precision', color='skyblue')
    plt.bar(x + width / 2, results["Recall"], width, label='Recall', color='lightgreen')
    plt.xticks(x, results["Model"])
    plt.title('Precision and Recall Comparison')
    plt.ylim(0, 1)
    plt.legend()

    plt.tight_layout()
    plt.savefig(args.output_fig)
    print(f"结果图表已保存至 {args.output_fig}")

    # 关闭Spark会话
    spark.stop()


if __name__ == "__main__":
    main()