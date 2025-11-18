import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CelebADataset(Dataset):
    def __init__(
            self, 
            celebA_image_path: str, 
            celebA_attr_path: str, 
            transform=None, 
            filter_attr=None, 
            filter_value=None, 
            max_samples=None
        ):
        self.celebA_image_path = celebA_image_path
        self.celebA_attr_path = celebA_attr_path
        self.transform = transform
        self.filter_attr = filter_attr
        self.filter_value = filter_value
        self.max_samples = max_samples
        
        self.attr_df = None
        self.image_list = None
        
        self.attr_df = pd.read_csv(celebA_attr_path)
        if 'image_id' not in self.attr_df.columns and len(self.attr_df.columns) > 0:
            self.attr_df.columns = ['image_id'] + list(self.attr_df.columns[1:])
        
        if filter_attr and filter_attr in self.attr_df.columns:
            filtered = self.attr_df[self.attr_df[filter_attr] == 1 if filter_value == 1 else -1]
            self.image_list = filtered['image_id'].tolist()
        else:
            self.image_list = self.attr_df['image_id'].tolist()

        if self.max_samples:
            self.image_list = self.image_list[:self.max_samples]
        
        print(f"[Dataset] CelebADataset __init__ success ({len(self.image_list)})")
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.celebA_image_path, img_name)
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name
    

def get_celeba_loader(
        celebA_image_path: str, 
        celebA_attr_path: str, 
        batch_size=64, 
        image_size=64,
        filter_attr=None, 
        filter_value=None,
        max_samples=None
    ):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    dataset = CelebADataset(
        celebA_image_path=celebA_image_path,
        celebA_attr_path=celebA_attr_path,
        transform=transform,
        filter_attr=filter_attr,
        filter_value=filter_value,
        max_samples=max_samples
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"[Dataset] get_celeba_loader success ({celebA_image_path}, {celebA_attr_path})")
    return dataloader

