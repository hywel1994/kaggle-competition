import os
import glob
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from progressbar import *

class ImageSet_preprocess(Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, item):
        image_path_1 = self.df.iloc[item]['image_path']
        image_path_2 = image_path_1.replace('/IJCAI_2019_AAAC_train/', '/IJCAI_2019_AAAC_train_processed/')
        _dir, _filename = os.path.split(image_path_2)
        
        if _filename == '3fece42d41e5f4d21edadc0e04cfba7e.jpg':
            print (image_path_2)
        if os.path.exists(image_path_2) and os.path.getsize(image_path_2)!=0:
            return image_path_2
        else:
            image = Image.open(image_path_1).convert('RGB')
            
            if not os.path.exists(_dir):
                os.makedirs(_dir)
            image.save(image_path_2)
            return image_path_2
    
def load_data_jpeg_compression(batch_size=16):
    all_imgs = glob.glob('/data/IJCAI_2019_AAAC_train/*/*.jpg')
    train_data = pd.DataFrame({'image_path':all_imgs})
    datasets = {
        'train_data': ImageSet_preprocess(train_data),
    }
    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       num_workers=8,
                       shuffle=False) for ds in datasets.keys()
    }
    return dataloaders

if __name__ == '__main__':

    dataloader = load_data_jpeg_compression()
    widgets = ['jpeg :',Percentage(), ' ', Bar('#'),' ', Timer(),
       ' ', ETA(), ' ', FileTransferSpeed()]
    pbar = ProgressBar(widgets=widgets)
    for batch_data in pbar(dataloader['train_data']):
        pass
