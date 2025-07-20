import matplotlib.pyplot as plt
import os
import csv

def save_training_results(model_name, learning_rate, batch_size, train_loss, val_loss, miou, iou_per_class, augmentation, save_dir="results/"):
    
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "training_results.csv")

    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            header = ["Model", "Learning Rate", "Batch Size", "Train Loss", "Validation Loss", "mIoU", "Augmentation", "Class", "IoU"]
            writer.writerow(header)

        for class_idx, iou in enumerate(iou_per_class):
            row = [
                model_name,
                float(learning_rate),
                int(batch_size),
                float(train_loss),
                float(val_loss),
                float(miou),
                bool(augmentation),
                f"Class_{class_idx + 1}",  
                float(iou) 
            ]
            writer.writerow(row)


def plot_training_results(train_losses, val_losses, miou_scores, num_classes, save_dir="results/"):
    epochs = range(1, len(train_losses) + 1)
    os.makedirs(save_dir, exist_ok=True)

    # train loss e validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train Loss vs Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "train_vs_val_loss.png"))
    plt.show()

    # mIoU
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, miou_scores, label="mIoU", color="green")
    plt.xlabel("Epochs")
    plt.ylabel("mIoU")
    plt.title("Mean IoU (mIoU) per Epoch")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "miou_per_epoch.png"))
    plt.show()