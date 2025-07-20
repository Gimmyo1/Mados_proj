import torch
import numpy as np
from dataset import MadosDataset
from torch.utils.data import DataLoader
from unet import UNet
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from results import visualize_prediction

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
            #print(f"Processing batch {batch_num}/{len(dataloader_test)}")
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            outputs_original = outputs  
            labels_original = labels 

            
            outputs = outputs.permute(0, 2, 3, 1).reshape(-1, outputs.shape[1])
            labels = labels.reshape(-1)
            mask = labels != -1
            labels = labels[mask]
            outputs = outputs[mask]



            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            labels = labels.cpu().numpy()
            if (batch_num==49):
                visualize_prediction(prediction=torch.argmax(outputs_original[0], dim=0).cpu().numpy())

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








if __name__ == "__main__":
    
    dataset_test = MadosDataset(mode='test')
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    
    model =  UNet(in_channels=4, out_channels=15)

    model_path = r"best_model.pth"
    
    model.load_state_dict(torch.load(model_path))
    print("Modello caricato.")

    
    test_loss, precision, recall, f1, miou = test_unet(model, dataloader_test)
    print(f"Test completato. Loss: {test_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, mIoU: {miou:.4f}")