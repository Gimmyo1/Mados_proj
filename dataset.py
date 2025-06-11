import torch
from torch.utils.data import Dataset
import rasterio
import os
import numpy as np
import torchvision.transforms as transforms
root_dir = "d:/Utente/Desktop/Mados_proj/data/MADOS"
split_dir = "d:/Utente/Desktop/Mados_proj/data/MADOS/splits"

class_distr = torch.Tensor([0.00336, 0.00241, 0.00336, 0.00142, 0.00775, 0.18452, 
 0.34775, 0.20638, 0.00062, 0.1169, 0.09188, 0.01309, 0.00917, 0.00176, 0.00963])


bands_mean = np.array([0.05223386, 0.04381474, 0.0357083, 0.03566642]).astype('float32')

bands_std = np.array([0.03432253, 0.0354812,  0.0375769, 0.05545856]).astype('float32')


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

        label_path = self.label_files[idx]
        label = rasterio.open(label_path).read(1).astype(np.int64)  # Carica la label come array 2D

        #label = np.where(np.isin(label, [1, 6]), 1, 0).float()
        label= label-1
        #print("Media dei dati normalizzati:", np.mean(img))
        #print("Deviazione standard dei dati normalizzati:", np.std(img))
       

        return img, label

def gen_weights(class_distribution, c = 1.02):
    return 1/torch.log(c + class_distribution)
	


dataset = MadosDataset(mode='train')

print("Numero di campioni:", len(dataset))

# Prendi il primo elemento
image, label = dataset[50]
print("Shape immagine:", image.shape)
print("Shape label:", label.shape)
print("Valori unici nella label:", set(label.flatten()))
print("img",label)
print("Valori unici nella label:", np.unique(label))


#print("Media dei dati normalizzati:", np.mean(image))
#print("Deviazione standard dei dati normalizzati:", np.std(image))

'''

image_files = dataset.image_files  # Lista di immagini multi-banda

# Inizializza array per accumulare i valori dei pixel
num_bands = 4  # Numero di bande che stai usando
pixel_sums = np.zeros(num_bands, dtype=np.float64)
pixel_squared_sums = np.zeros(num_bands, dtype=np.float64)
pixel_counts = np.zeros(num_bands, dtype=np.int64)

for band_paths in image_files:
    bands = [rasterio.open(band_path).read(1) for band_path in band_paths]
    img = np.stack(bands, axis=0)  # Forma (num_bande, altezza, larghezza)

    # Ignora i valori NaN durante il calcolo
    pixel_sums += np.nansum(img, axis=(1, 2))  # Somma dei pixel ignorando i NaN
    pixel_squared_sums += np.nansum(img ** 2, axis=(1, 2))  # Somma dei quadrati dei pixel ignorando i NaN
    pixel_counts += np.sum(~np.isnan(img), axis=(1, 2))  # Conta i pixel validi (non NaN)

# Calcola la media per ciascuna banda ignorando i NaN
bands_mean = pixel_sums / pixel_counts

# Calcola la deviazione standard per ciascuna banda ignorando i NaN
bands_std = np.sqrt(pixel_squared_sums / pixel_counts - bands_mean ** 2)

print("Media per banda:", bands_mean)
print("Deviazione standard per banda:", bands_std)
'''