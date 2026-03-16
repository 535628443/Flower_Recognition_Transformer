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