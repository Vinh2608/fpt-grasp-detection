from torch.utils.data import Dataset, DataLoader
import torch
import clip 
from PIL import Image
import os
import pickle

def load_instructions_from_pkl(file_path):
    with open(file_path, 'rb') as file:
        # Load the data from the file
        data = pickle.load(file)
    return data

class Grasp_Dataset(Dataset):
    def __init__(self, directory_image, directory_instructions, directory_labels, device):
        self.directory_image = directory_image
        self.directory_instructions = directory_instructions
        self.directory_labels = directory_labels

        self.image_paths = os.listdir(self.directory_image)
        self.instruction_paths = os.listdir(self.directory_instructions)
        self.labels_paths = os.listdir(self.directory_labels)

        self.ids = []
        
        for image_file in self.image_paths:
            base_name = os.path.splitext(image_file)[0]  # Remove the extension from image file
            # Find corresponding instruction file
            instruction_file = base_name + '.pkl'  # Assuming instruction files are .pkl files
            labels_file = base_name + '.pt'
            if instruction_file in self.instruction_paths and labels_file in self.labels_paths:
                self.ids.append((image_file, instruction_file, labels_file)) 
        
        _, self.preprocess = clip.load("ViT-B/32", device=device)
        print('ok')
        self.device = device

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        full_img_path = os.path.join(self.directory_image,self.ids[idx][0])
        full_instruction_path = os.path.join(self.directory_instructions,self.ids[idx][1])
        full_label_path = os.path.join(self.directory_labels,self.ids[idx][2])

        image = self.preprocess(Image.open(full_img_path)).unsqueeze(0).to(self.device)
        text = load_instructions_from_pkl(full_instruction_path)
        
        text = clip.tokenize(text).to(self.device)
        label = torch.load(full_label_path).cpu().numpy()
        return image, text, label

# Example usage

