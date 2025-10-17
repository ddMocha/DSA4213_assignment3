import os, json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def save_training_curves(trainer, out_dir):
    # Trainer 会在 state.log_history 里写入每个 epoch 的指标
    logs = [x for x in trainer.state.log_history if "loss" in x or "eval_loss" in x]
    epochs = sorted({x["epoch"] for x in logs if "epoch" in x})
    train_loss = [x["loss"] for x in logs if "loss" in x and "epoch" in x]
    eval_loss  = [x["eval_loss"] for x in logs if "eval_loss" in x and "epoch" in x]
    # 简化画法：按出现顺序近似对应 epoch
    plt.figure()
    plt.plot(train_loss, label="train_loss")
    plt.plot(eval_loss, label="eval_loss")
    plt.xlabel("Epoch (approx)"); plt.ylabel("Loss"); plt.legend()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "loss_curves.png"), bbox_inches="tight"); plt.close()

def save_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix"); plt.colorbar()
    tick = np.arange(cm.shape[0])
    plt.xticks(tick, tick); plt.yticks(tick, tick)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.savefig(out_path, bbox_inches="tight"); plt.close()
