import torch
from dataset import MadosDataset, gen_weights, class_distr
from torch.utils.data import DataLoader
from unet import UNet
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

def test_unet(model, dataloader_test):
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  

    weight = gen_weights(class_distr, c = 1.03)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='mean', weight=weight.to(device))

    #criterion = torch.nn.CrossEntropyLoss()
    test_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad(): 
        for images, labels in dataloader_test:
            images = images.to(device)
            labels = labels.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # predictions
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy().flatten())
            y_pred.extend(preds.cpu().numpy().flatten())

    avg_test_loss = test_loss / len(dataloader_test)
    print(f"Test Loss: {avg_test_loss:.4f}")

    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)

    # Precision, Recall, F1-score
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Iou
    iou = jaccard_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"Test Accuracy: {(y_true == y_pred).sum().item() / len(y_true):.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"IoU: {iou:.4f}")

    return avg_test_loss, precision, recall, f1, iou


if __name__ == "__main__":
    
    dataset_test = MadosDataset(mode='test')
    dataloader_test = DataLoader(dataset_test, batch_size=4, shuffle=False)

    
    model = UNet(in_channels=4, out_channels=16)

    
    model.load_state_dict(torch.load("unet_mados.pth"))
    print("Modello caricato.")

    
    test_loss, precision, recall, f1, iou = test_unet(model, dataloader_test)
    print(f"Test completato. Loss: {test_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, IoU: {iou:.4f}")