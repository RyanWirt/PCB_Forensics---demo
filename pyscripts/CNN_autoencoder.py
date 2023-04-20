import torch.nn as nn
from torchvision import transforms

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""



class CNNAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()        
        # B, 3, 400, 400
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 2, stride = 2, padding = 6),
            nn.ReLU(),
            nn.Conv2d(64, 192, 5, stride = 2, padding = 2),
            nn.ReLU(),
            nn.Conv2d(192, 192, 5, stride = 2, padding = 0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(192, 300, 5, stride = 2, padding = 0),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(300, 64, 5, stride = 2, padding = 0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 5, stride = 2, padding = 1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(64, 32, 5, stride = 1, padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 2, stride = 2, padding = 0),
            nn.Sigmoid()
        )

    def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            transform = transforms.Resize(size = (400,400))
            x = transform(decoded)
            return x
