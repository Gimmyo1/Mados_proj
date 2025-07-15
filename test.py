import torch
import numpy as np
from dataset import MadosDataset, gen_weights, class_distr
from torch.utils.data import DataLoader
from unet import UNet
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

def test_unet(model, dataloader_test):
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  

    weight = gen_weights(class_distr, c = 1.03)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')

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

            
            outputs = outputs.permute(0, 2, 3, 1).reshape(-1, outputs.shape[1])
            labels = labels.reshape(-1)
            mask = labels != -1
            labels = labels[mask]
            outputs = outputs[mask]



            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            labels = labels.cpu().numpy()

            y_pred += preds.tolist()
            y_true += labels.tolist()

    avg_test_loss = test_loss / len(dataloader_test)
    print(f"Test Loss: {avg_test_loss:.4f}")

    #y_true = torch.tensor(y_true)
    #y_pred = torch.tensor(y_pred)

    # Metriche
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # mIou
    miou = jaccard_score(y_true, y_pred, average='macro', zero_division=0)
    accuracy = (np.array(y_true) == np.array(y_pred)).mean()

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"mIoU: {miou:.4f}")

    return avg_test_loss, precision, recall, f1, miou


if __name__ == "__main__":
    
    dataset_test = MadosDataset(mode='test')
    dataloader_test = DataLoader(dataset_test, batch_size=4, shuffle=False)

    
    model =  UNet(in_channels=4, out_channels=15)

    
    model.load_state_dict(torch.load("best_model.pth"))
    print("Modello caricato.")

    
    test_loss, precision, recall, f1, miou = test_unet(model, dataloader_test)
    print(f"Test completato. Loss: {test_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, mIoU: {miou:.4f}")