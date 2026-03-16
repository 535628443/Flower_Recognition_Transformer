import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import os

def save_model(model, target_dir, model_name):
    """保存 PyTorch 模型"""
    os.makedirs(target_dir, exist_ok=True)
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "模型名称必须以 '.pt' 或 '.pth' 结尾"
    model_save_path = os.path.join(target_dir, model_name)
    print(f"[INFO] 正在保存模型至: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

def plot_loss_curves(results):
    """
    绘制训练和验证的 Loss 和 Accuracy 曲线。
    results 是一个字典，包含 'train_loss', 'train_acc', 'val_loss', 'val_acc'
    """
    loss = results['train_loss']
    test_loss = results['val_loss']
    accuracy = results['train_acc']
    test_accuracy = results['val_acc']

    epochs = range(len(results['train_loss']))

    plt.figure(figsize=(15, 7))

    # 绘制 Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss', marker='o')
    plt.plot(epochs, test_loss, label='val_loss', marker='o')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制 Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy', marker='o')
    plt.plot(epochs, test_accuracy, label='val_accuracy', marker='o')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 确保保存目录存在
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/loss_curves.png')
    print("[INFO] 训练曲线已保存至 results/loss_curves.png")

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    绘制混淆矩阵并保存
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/confusion_matrix.png')
    print("[INFO] 混淆矩阵已保存至 results/confusion_matrix.png")
