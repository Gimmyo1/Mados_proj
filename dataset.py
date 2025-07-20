import torch
from torch.utils.data import Dataset
import rasterio
import os
import numpy as np
import torchvision.transforms as transforms
import albumentations as A
import matplotlib.pyplot as plt

root_dir = "d:/Utente/Desktop/Mados_proj/data/MADOS"
split_dir = "d:/Utente/Desktop/Mados_proj/data/MADOS/splits"

# Distribuzione delle classi normalizzata
class_distr = torch.Tensor([0.0034, 0.0024, 0.0034, 0.0014, 0.0078, 0.1845, 
 0.3478, 0.2064, 0.0006, 0.117, 0.0919, 0.0131, 0.0092, 0.0018, 0.0096])



bands_mean = np.array([0.05223386, 0.04381474, 0.0357083, 0.03566642]).astype('float32')

bands_std = np.array([0.03432253, 0.0354812,  0.0375769, 0.05545856]).astype('float32')


def get_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5)
    ], p=1)


class MadosDataset(Dataset):
    def __init__(self, root_dir=root_dir, splits_dir=split_dir, resolution='10', transform=None , mode='train'):
        self.root_dir = root_dir
        self.splits_dir = splits_dir
        self.resolution = resolution
        self.transform = transform
        self.mode = mode
        self.normalize = transforms.Normalize(mean=bands_mean, std=bands_std)
        self.image_files = []
        self.label_files = []
        #self.impute_nan = np.tile(bands_mean, (self.image_files.shape[-1],self.image_files.shape[-2],1))
        ignorati = 0

        if mode == 'train':
            split_file = os.path.join(splits_dir, 'train_X.txt')
            self.transform = get_transform()
        elif mode == 'val':
            split_file = os.path.join(splits_dir, 'val_X.txt')
        elif mode == 'test':
            split_file = os.path.join(splits_dir, 'test_X.txt')
        else:
            raise ValueError(f"Modalità non valida: {mode}. Usa 'train', 'val' o 'test'.")


        with open(split_file, 'r') as f:
            patch_list = [line.strip() for line in f.readlines()]

        for patch in patch_list:
            base_name, suffix = patch.rsplit('_', 1)  

            # path
            label_path = os.path.join(
                self.root_dir, base_name, self.resolution, f"{base_name}_L2R_cl_{suffix}.tif"
            )
            band_paths = [
                os.path.join(
                    self.root_dir, base_name, self.resolution, f"{base_name}_L2R_rhorc_492_{suffix}.tif"
                ),
                os.path.join(
                    self.root_dir, base_name, self.resolution, f"{base_name}_L2R_rhorc_560_{suffix}.tif"
                ),
                os.path.join(
                    self.root_dir, base_name, self.resolution, f"{base_name}_L2R_rhorc_665_{suffix}.tif"
                ),
                os.path.join(
                    self.root_dir, base_name, self.resolution, f"{base_name}_L2R_rhorc_833_{suffix}.tif"
                ),
            ]
            
            # Verifica esistenza file
            if all(os.path.exists(b) for b in band_paths) and os.path.exists(label_path):
                self.image_files.append(band_paths)
                self.label_files.append(label_path)
            else:
                ignorati+=1
                print(f"Crop ignorato: {patch}")
                print(f"Label path: {label_path} - Esiste: {os.path.exists(label_path)}")
                for b in band_paths:
                    print(f"Band path: {b} - Esiste: {os.path.exists(b)}")
        print(f"Totale crop ignorati: {ignorati}")


    def __len__(self):
        return len(self.image_files)
    

    def __getitem__(self, idx):
        band_paths = self.image_files[idx]
        bands = [rasterio.open(band_path).read(1) for band_path in band_paths] 
        img = np.stack(bands, axis=0).astype(np.float32)  # bande in un array 4x240x240

       # nan_mask = np.isnan(img)
        #print("Numero di NaN trovati:", np.sum(nan_mask))

        img = np.nan_to_num(img, nan=bands_mean[:, None, None])

        # Normalizzazione
        #img = (img - bands_mean[:, None, None]) / bands_std[:, None, None]        #mean_band = np.mean(img, axis=(1, 2))
        img = self.normalize(torch.tensor(img))
        #std_band = np.std(img, axis=(1, 2))
        #img = (img - mean_band) / self.std_band
        #img = torch.tensor(img)
        label_path = self.label_files[idx]
        label = rasterio.open(label_path).read(1).astype(np.int64)  # Carica la label come array 2D
        label= label-1
        #print("Media dei dati normalizzati:", np.mean(img))
        #print("Deviazione standard dei dati normalizzati:", np.std(img))
        if self.transform:
        
            augmented = self.transform(image=img.numpy().transpose(1, 2, 0), mask=label)
            img = augmented["image"].transpose(2, 0, 1) 
            label = augmented["mask"].astype(np.int64) 

        #img=torch.from_numpy(img)
        #label=torch.from_numpy(label)

        return img, label

def gen_weights(class_distribution, c = 1.02):
    return 1/torch.log(c + class_distribution)
	


# DEBUGGING

dataset = MadosDataset(mode='train')

print("Numero di campioni:", len(dataset))

# Prendi il primo elemento
image, label = dataset[923]
#print("Shape immagine:", image.shape)
#print("Shape label:", label.shape)
#print("Valori unici nella label:", set(label.flatten()))
#print("img",image.max(), image.min())
#print("Valori unici nella label:", np.unique(label))

#print("Media dei dati normalizzati:", np.mean(image))
#print("Deviazione standard dei dati normalizzati:", np.std(image))



def multispectral_to_rgb_visualization(img, lower_percentile=5, upper_percentile=95):


    assert isinstance(img, np.ndarray), "The input image must be a numpy array"
    img = img.transpose(1,2,0)
    img = img[:, :, [2, 1, 0]]
    img = np.clip(img, np.percentile(img, lower_percentile), np.percentile(img, upper_percentile))
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = (img * 255).astype(np.uint8)
    plt.imshow(img)
    plt.axis('off') 
    plt.title("Multispectral Image Visualization")
    plt.show()
    return img

#img= multispectral_to_rgb_visualization(image)
#print(img)

