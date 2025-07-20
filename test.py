import torch
import numpy as np
from dataset import MadosDataset
from torch.utils.data import DataLoader
from unet import UNet
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

def test_unet(model, dataloader_test):
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')

    test_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        batch_num = 0 
        for images, labels in dataloader_test:
            images = images.to(device)
            labels = labels.to(device)
            batch_num += 1
            print(f"Processing batch {batch_num}/{len(dataloader_test)}")
            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            outputs_original = outputs  
            labels_original = labels 

            # Reshape outputs e labels
            outputs = outputs.permute(0, 2, 3, 1).reshape(-1, outputs.shape[1])
            labels = labels.reshape(-1)
            mask = labels != -1
            labels = labels[mask]
            outputs = outputs[mask]



            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            labels = labels.cpu().numpy()
            #if (batch_num==49):
            #    print(labels_original[0].cpu().numpy())
            #    visualize_prediction(
            #    image=images[0],  # Immagine originale
            #    ground_truth=labels_original[0].cpu().numpy(),  # Ground truth originale
            #    prediction=torch.argmax(outputs_original[0], dim=0).cpu().numpy())

            y_pred += preds.tolist()
            y_true += labels.tolist()

    avg_test_loss = test_loss / len(dataloader_test)
    print(f"Test Loss: {avg_test_loss:.4f}")
    

    num_classes = 15  
    iou_per_class = jaccard_score(y_true, y_pred, average=None, labels=range(num_classes))

    for i, iou in enumerate(iou_per_class):
        print(f"Classe {i+1}: IoU = {iou:.4f}")

    #y_true = torch.tensor(y_true)
    #y_pred = torch.tensor(y_pred)

    # metriche
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    miou = jaccard_score(y_true, y_pred, average='macro', zero_division=0)
    accuracy = (np.array(y_true) == np.array(y_pred)).mean()

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"mIoU: {miou:.4f}")

    return avg_test_loss, precision, recall, f1, miou




import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

def visualize_prediction(image, ground_truth, prediction):

    class_labels = [
        "Marine Debris", "Dense Sargassum", "Sparse Floating Algae", "Natural Organic Material",
        "Ship", "Oil Spill", "Marine Water", "Sediment-Laden Water", "Foam", "Turbid Water",
        "Shallow Water", "Waves & Wakes", "Oil Platform", "Jellyfish", "Sea snot"
    ]

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5"
    ]

    cmap = ListedColormap(colors[:len(class_labels)])
    norm = BoundaryNorm(range(len(class_labels) + 1), cmap.N)

    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()  

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image.astype('uint8'))  # Assicurati che l'immagine sia in formato uint8
    plt.title("Immagine Originale")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    gt = plt.imshow(ground_truth, cmap=cmap, norm=norm)
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    pred = plt.imshow(prediction, cmap=cmap, norm=norm)
    plt.title("Predizione")
    plt.axis("off")

    cbar = plt.colorbar(gt, ax=plt.gcf().axes, orientation='horizontal', pad=0.05, aspect=50)
    cbar.set_ticks(range(len(class_labels)))
    cbar.set_ticklabels(class_labels)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    dataset_test = MadosDataset(mode='test')
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    
    model =  UNet(in_channels=4, out_channels=15)

    model_path = r"results\Model_3\best_model.pth"
    
    model.load_state_dict(torch.load(model_path))
    print("Modello caricato.")

    
    test_loss, precision, recall, f1, miou = test_unet(model, dataloader_test)
    print(f"Test completato. Loss: {test_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, mIoU: {miou:.4f}")