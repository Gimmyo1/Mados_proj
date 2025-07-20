import torch
import torch.nn as nn
import numpy as np
import os
import torch.optim as optim
from unet import UNet
from dataset import MadosDataset,class_weights, class_distr
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from results import plot_training_results, save_training_results

def train_unet(model, dataloader_train, dataloader_val, num_epochs=10, learning_rate=0.001, save_dir="results/"):


    

    
    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    weight = class_weights(class_distr)
    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean', weight=weight.to(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_miou = 0
    train_losses = []
    val_losses = []
    miou_scores = []
    iou_per_class_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in dataloader_train:

            images = images.to(device)
            labels = labels.to(device) 

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch Loss: {loss.item():.4f}")
        avg_loss = running_loss / len(dataloader_train)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")


        model.eval()
        best_epoch = 0
        val_loss = 0.0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for images, labels in dataloader_val:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # scarta i pixel non annotati
                outputs = outputs.permute(0, 2, 3, 1).reshape(-1, outputs.shape[1])
                labels = labels.reshape(-1)
                mask = labels != -1

                outputs = outputs[mask]
                labels = labels[mask]

                probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                labels = labels.cpu().numpy()

                y_pred += preds.tolist()
                y_true += labels.tolist()

        avg_val_loss = val_loss / len(dataloader_val)
        val_losses.append(avg_val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

        
        num_classes = 15
        iou_per_class = jaccard_score(y_true, y_pred, average=None, labels=range(num_classes))
        miou = jaccard_score(y_true, y_pred, average='macro', zero_division=0)
        miou_scores.append(miou)
        iou_per_class_history.append(iou_per_class)


        print(f"Epoch [{epoch+1}/{num_epochs}], Validation mIoU: {miou:.4f}")

       
        if miou > best_miou:
            best_miou = miou
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print("miglior modello salvato")


        

    
    
    print("training finito")
    return model, train_losses, val_losses, miou_scores,iou_per_class_history, best_epoch




if __name__ == "__main__":


    experiment_id = "Model_3"
    save_dir = f"results/{experiment_id}/"
    os.makedirs(save_dir, exist_ok=True)
    
    learning_rate = 0.0001
    num_epochs = 40
    batch_size = 5
    augmentation = True

    
    dataset_train = MadosDataset(mode='train')
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    dataset_val = MadosDataset(mode='val')
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    model =  UNet(in_channels=4, out_channels=15)
    
  

    
    trained_model, train_losses, val_losses, miou_scores,iou_per_class, best_epoch = train_unet(model, dataloader_train, dataloader_val, num_epochs=num_epochs, learning_rate=learning_rate, save_dir=save_dir)
    
    
    
    #torch.save(trained_model.state_dict(), "unet_mados.pth")

    plot_training_results(train_losses, val_losses, miou_scores, num_classes=15, save_dir=save_dir)

    save_training_results(
        model_name="UNet",
        learning_rate=learning_rate,
        batch_size=batch_size,
        train_loss=train_losses[best_epoch-1],
        val_loss=val_losses[best_epoch-1],
        miou=miou_scores[best_epoch-1],
        iou_per_class=iou_per_class[best_epoch-1],
        augmentation=augmentation,
        save_dir=save_dir
    )