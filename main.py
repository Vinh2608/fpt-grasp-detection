from model import ClipPredictor
import clip
from torch import nn, optim
from dataset import Grasp_Dataset
import tqdm
import argparse
import os
from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description='')

    # basic settings
    parser.add_argument('--data-train-root', type=str, required=True)
    parser.add_argument('--data-val-root', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    
    # semi-supervised settings
    parser.add_argument('--save-path', type=str, required=True)

    # arguments for ST++

    args = parser.parse_args()
    return args

def main(args):
    train_image_path = args.data_train_root + '/image'
    train_instructions_path = args.data_train_root + '/grasp_instructions'
    train_labels_path = args.data_train_root + '/grasp_label'

    val_image_path = args.data_val_root + '/image'
    val_instructions_path = args.data_val_root + '/grasp_instructions'
    val_labels_path = args.data_val_root + '/grasp_label'

    device = "cuda:0"

    train_dataset = Grasp_Dataset(train_image_path, train_instructions_path, train_labels_path, device)
    val_dataset = Grasp_Dataset(val_image_path, val_instructions_path, val_labels_path, device)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    device = 'cuda'
    model, preprocess = clip.load("ViT-B/16", device=device)

    model = ClipPredictor(model).to(device)
    optimizer = optim.Adam(model.regressor.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    train(args.epoch, train_dataloader, val_dataloader, optimizer, model, criterion, device, args)

def pad_labels(labels, max_length=36, pad_value=0):
    padded_labels = pad_value * np.ones((max_length, 6))
    padded_labels[:len(labels)] = labels
    mask = np.zeros((max_length, 6))
    mask[:len(labels)] = 1
    return padded_labels, mask

def validate(val_dataloader, model, criterion, device):
    
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():  # No gradients needed for validation
        for images, texts, labels in tqdm(val_dataloader):
            images = images.to(device)
            texts = texts.to(device)
            outputs = model(images, texts)
            labels, mask = pad_labels(labels)  # Assume pad_labels returns torch tensors
            labels, mask = labels.to(device), mask.to(device)
            loss = criterion(outputs, labels)
            loss = (loss * mask).mean()  # Average the loss after applying the mask
            total_loss += loss.item()
    
    average_loss = total_loss / len(val_dataloader)
    print(f"Validation Loss: {average_loss}")
    return average_loss

def train(epochs, train_dataloader, val_dataloader, optimizer, model, criterion, device, args):
    best_loss = float('inf')

    for epoch in range(epochs):  # number of epochs
        for images, texts, labels in tqdm(train_dataloader):
            optimizer.zero_grad()
            outputs = model(images, texts)
            labels, mask = pad_labels(labels)
            loss = criterion(outputs, labels.to(device))
            loss = loss * mask
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")

        val_loss = validate(val_dataloader, model, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.save_path, f'model_epoch_{epoch+1}_loss_{val_loss:.4f}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == '__main__':
    args = parse_args()


    print()
    print(args)

    main(args)
