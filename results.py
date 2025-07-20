import matplotlib.pyplot as plt
import os
import csv
from matplotlib.colors import ListedColormap, BoundaryNorm

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




def visualize_prediction(prediction,save_dir="prediction.png"):

    class_labels = ["1", "2", "3", "4","5", "6", "7", "8", "9", "10","11", "12", "13", "14", "15"]

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5"
    ]

    cmap = ListedColormap(colors[:len(class_labels)])
    norm = BoundaryNorm(range(len(class_labels) + 1), cmap.N)


    plt.figure(figsize=(6, 5))

    pred = plt.imshow(prediction, cmap=cmap, norm=norm)
    plt.title("Predizione")
    plt.axis("off")

    cbar = plt.colorbar(ax=plt.gcf().axes, orientation='horizontal', pad=0.5, aspect=50)
    cbar.set_ticks(range(len(class_labels)))
    cbar.set_ticklabels(class_labels)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.show()
