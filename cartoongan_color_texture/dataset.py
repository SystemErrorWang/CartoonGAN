import cv2
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

    
class Dataset(Dataset):


    def __init__(self, folder, mode='train', transform=None):
        self.mode = mode
        self.transform = transform
        self.image_list, self.label_list = list(), list()
        for sub_folder in os.listdir(folder):
            if mode == 'train':
                image_folder = os.path.join(folder, sub_folder, 
                                            'training_data', 'cellphone')
                for name in os.listdir(image_folder):
                    image_path = os.path.join(image_folder, name)
                    label_path = image_path.replace('cellphone', 'canon')
                    self.image_list.append(image_path)
                    self.label_list.append(label_path)
                
            elif mode == 'test':
                image_folder = os.path.join(folder, sub_folder, 
                                            'test_data', 'full_size_test_images')
                for name in os.listdir(image_folder):
                    image_path = os.path.join(image_folder, name)
                    #label_path = image_path.replace('cellphone', 'canon')
                    self.image_list.append(image_path)
                    #self.label_list.append(label_path)
            
            

    def __len__(self):
        return len(self.image_list)


    def __getitem__(self, idx):
        if self.mode == 'train':
            image = cv2.imread(self.image_list[idx])
            label = cv2.imread(self.label_list[idx])
            
            if self.transform is not None:
                image, label = self.transform(image, label)
            
            return image, label
        
        elif self.mode == 'test':
            image = cv2.imread(self.image_list[idx])
            
            if self.transform is not None:
                image = self.transform(image)

            return image

        
class CartoonDataset(Dataset):


    def __init__(self, photo_dir, cartoon_dir, transform=None):
        self.transform = transform
        self.photo_list, self.cartoon_list = list(), list()
        #image_folder = os.path.join(data_dir, 'dped')
        #label_folder = os.path.join(data_dir, 'cartoon_dataset')
        for name in os.listdir(photo_dir):
            photo_path = os.path.join(photo_dir, name)
            self.photo_list.append(photo_path)
                
                
        for name in os.listdir(cartoon_dir):
            cartoon_path = os.path.join(cartoon_dir, name)
            self.cartoon_list.append(cartoon_path)
            

    def __len__(self):
        return min(len(self.photo_list), len(self.cartoon_list))


    def __getitem__(self, idx):
        np.random.shuffle(self.cartoon_list)
        photo = cv2.imread(self.photo_list[idx])
        cartoon = cv2.imread(self.cartoon_list[idx])
        
        if self.transform is not None:
            photo, cartoon = self.transform(photo, cartoon)
        
        return photo, cartoon
        
       

class CartoonTransform():
    
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        
        
    def __call__(self, image, label):
        
        new_h, new_w = self.output_size
        image_h, image_w = np.shape(image)[:2]
        label_h, label_w = np.shape(label)[:2]
        image_dh = np.random.randint(0, image_h - new_h)
        image_dw = np.random.randint(0, image_w - new_w)
        label_dh = np.random.randint(0, label_h - new_h)
        label_dw = np.random.randint(0, label_w - new_w)

        image = image[image_dh: image_dh + new_h, 
                      image_dw: image_dw + new_w]
        label = label[label_dh: label_dh + new_h, 
                      label_dw: label_dw + new_w]
                                      
        flip_prop = np.random.randint(0, 100)
        if flip_prop > 50:
            image = cv2.flip(image, 1)   
            label = cv2.flip(label, 1)   
        
        image = image.astype(np.float32) / 127.5 - 1
        label = label.astype(np.float32) / 127.5 - 1
        
        return image, label
    
    
class TrainTransform():
    
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        
        
    def __call__(self, image, label):
        
        new_h, new_w = self.output_size
        h, w = np.shape(image)[:2]
        offset_h = np.random.randint(0, h - new_h)
        offset_w = np.random.randint(0, w - new_w)

        image = image[offset_h: offset_h + new_h, 
                                offset_w: offset_w + new_w]
        label = label[offset_h: offset_h + new_h, 
                                offset_w: offset_w + new_w]
                                      
        flip_prop = np.random.randint(0, 100)
        if flip_prop > 50:
            image = cv2.flip(image, 1)   
            label = cv2.flip(label, 1)   
            
        rotate_prop = np.random.randint(0, 3)
        M1 = cv2.getRotationMatrix2D((new_h/2-0.5, new_w/2-0.5), 90*rotate_prop, 1)
        image = cv2.warpAffine(image, M1, (new_h, new_w))
        label = cv2.warpAffine(label, M1, (new_h, new_w))
        
        image = image.astype(np.float32) / 127.5 - 1
        label = label.astype(np.float32) / 127.5 - 1
        
        return image, label
    
    
    
class TestTransform():
    
    def __init__(self):
        pass
        
        
    def __call__(self, image):
        
        image = image.astype(np.float32)/127.5 - 1
        
        return image



def get_train_loader(image_size, batch_size, data_dir):
    transform = TrainTransform(image_size)
    dataset = Dataset(data_dir, mode='train', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=8)
    return dataloader


def cartoon_loader(image_size, batch_size, photo_dir, cartoon_dir):
    transform = CartoonTransform(image_size)
    dataset = CartoonDataset(photo_dir, cartoon_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=8)
    return dataloader


def get_test_loader(data_dir):
    transform = TestTransform()
    dataset = Dataset(data_dir, mode='test', transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, 
                            shuffle=False, num_workers=0)
    return dataloader



if __name__ == '__main__':
    '''
    data_dir = 'C:\\Users\\Razer\\Downloads\\dataset'
    #dataloader = get_train_loader(96, 16, data_dir) 
    dataloader = cartoon_loader(96, 16, data_dir) 

    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        pass
    '''
    cartoon_dir = 'C:\\Users\\Razer\\Downloads\\dataset\\shinkai_makoto'
    photo_dir = 'C:\\Users\\Razer\\Downloads\\dataset\\hr_photos'
    for name in tqdm(os.listdir(photo_dir)):
        file_path = os.path.join(photo_dir, name)
        try:
            image = cv2.imread(file_path)
            h, w, c = np.shape(image)
        except:
            os.remove(file_path)
