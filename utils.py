import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from pathlib import Path

# ---保存PyTorch模型--- 由于有多轮训练，每次都会保存最好的那版模型
def save_model(model, target_dir, model_name):
    # 确认目标目录存在(不存在就创建)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    # 检查文件后缀(模型一般以 .pth 或者 .pt 为尾缀)； assert 条件, "报错时的提示信息"
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "模型文件名必须以 .pth 或者 .pt 结尾"
    # 拼接完整的保存路径
    model_save_path = target_path / model_name
    print(f"[INFO] 正在保存模型到: {model_save_path}")
    # 保存
    torch.save(obj=model.state_dict(), f=model_save_path)

# 绘制loss和accuracy曲线
def plot_loss_curves(results):
    loss = results['train_loss']
    test_loss = results['val_loss']
    accuracy = results['train_acc']
    test_accuracy = results['val_acc']

    epochs = range(len(results['train_loss']))

    plt.figure(figsize=(15, 7))

    # 绘制 Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='val_loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()


    # 绘制 Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='val_accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
 
    # 确保保存目录存在
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/loss_curves.png')
    print("[INFO] 训练曲线已保存至 results/loss_curves.png")

def plot_confusion_matrix(y_true, y_pred, classes):
    """绘制混淆矩阵并保存"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/confusion_matrix.png')
    print("[INFO] 混淆矩阵已保存至 results/confusion_matrix.png")