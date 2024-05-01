import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn, optim

class ClipPredictor(nn.Module):
    def __init__(self, clip_model):
        super(ClipPredictor, self).__init__()
        self.clip_model = clip_model
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),  # Adjust the input size based on the CLIP model variant
            nn.ReLU(),
            nn.Linear(256, 36 * 6)    # Output the 6 parameters
        )

    def forward(self, image, text):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            text_features = self.clip_model.encode_text(text)

        # Concatenate the features (or you could also experiment with other fusion methods)
        features = torch.cat((image_features, text_features), dim=1)
        return self.regressor(features)