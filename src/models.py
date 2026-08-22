import torch
import torch.nn as nn
import numpy as np
import math

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, k, s, p)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, k=3, s=1, p=1):
        super().__init__()

        if not hidden_channels:
            hidden_channels = out_channels

        self.conv1 = ConvLayer(in_channels=in_channels, out_channels=hidden_channels, k=k, s=s, p=p)
        self.conv2 = ConvLayer(in_channels=hidden_channels, out_channels=out_channels, k=k, s=1, p=p)

        if in_channels != out_channels or s != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=s, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)

        x = self.conv1(x)
        x = self.conv2(x)

        return residual + x


class Encoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        layers = []
        out_channels = 32

        self.size = hparams.get("image_size", 28)
        latent_channels = hparams.get("latent_channels", 8)
        latent_dim = hparams.get("latent_dim", 16)
        input_channels = hparams.get("image_channels", 1)
        p = hparams.get("p", 0.2)

        num_downsamples = max(0, math.ceil(math.log2(self.size / 4)))

        for i in range(num_downsamples):
            if i == num_downsamples - 1: out_channels = latent_channels
            layers.extend([
                ResBlock(in_channels=input_channels, out_channels=out_channels, s=2),
                nn.Dropout(p),
            ])

            self.size = (self.size + 1) // 2

            input_channels = out_channels
            out_channels = min(out_channels * 2, 256)
        
        self.to_latent = nn.Linear(input_channels * self.size * self.size, latent_dim)

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten(1)
        return self.to_latent(x)


class Decoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.latent_channels = hparams.get("latent_channels", 8)
        latent_dim = hparams.get("latent_dim", 16)
        self.latent_size = hparams.get("latent_size", 4)
        out_channels = hparams.get("image_channels", 1)
        image_size = hparams.get("image_size", 28)

        self.from_latent = nn.Linear(latent_dim, self.latent_channels * self.latent_size * self.latent_size)

        sizes = self.get_sizes(self.latent_size, image_size)
        layers = []

        in_channels = self.latent_channels
        out_channels = 64

        for size in sizes[1:]:
            layers.extend([
                nn.Upsample(size=(size, size), mode="bilinear", align_corners=False),
                ResBlock(in_channels=in_channels, out_channels=out_channels),
            ])

            in_channels = out_channels
            out_channels = max(out_channels // 2, 16)
        
        layers.append(
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        )
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.from_latent(x)
        x = x.view(
            -1,
            self.latent_channels,
            self.latent_size,
            self.latent_size,
        )
        return self.decoder(x)

    def get_sizes(self, start, target):
        sizes = [start]

        while sizes[-1] < target:
            current = sizes[-1]

            # Roughly double the spatial resolution,
            # but never go beyond the target.
            next_size = min(current * 2, target)

            sizes.append(next_size)

        return sizes


class AutoEncoder(nn.Module):
    def __init__(self, hparams, encoder, decoder):
        super().__init__()

        self.hparams = hparams
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.encoder = encoder
        self.decoder = decoder
        self.set_optimizer()
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def training_step(self, batch, loss_func):
        self.encoder.train()
        self.decoder.train()

        self.optimizer.zero_grad()
        images = batch
        images = images.to(self.device)

        reconstruction = self(images)
        loss = loss_func(reconstruction, images)

        loss.backward()

        self.optimizer.step()
        return loss

    def validation_step(self, batch, loss_func):
        images = batch
        images = images.to(self.device)

        reconstruction = self(images)

        loss = loss_func(reconstruction, images)
        return loss

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.get("learning_rate", 1e-3))

    def getReconstructions(self, loader=None):

        assert loader is not None, "Please provide a dataloader for reconstruction"
        self.eval()
        self = self.to(self.device)

        reconstructions = []

        for batch in loader:
            X, _ = batch
            X = X.to(self.device)
            reconstruction = self.forward(X)
            reconstructions.append(
                reconstruction.view(-1, 28, 28).cpu().detach().numpy())

        return np.concatenate(reconstructions, axis=0)


class Classifier(nn.Module):
    def __init__(self, hparams, encoder):
        super().__init__()

        self.hparams = hparams
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        input_size = hparams.get("latent_dim", 16)

        log2_input = int(np.ceil(np.log2(input_size)))
        sizes = [input_size, 2 ** (log2_input + 1), 2 ** (log2_input + 2), hparams.get("num_classes", 10)]

        layers = []
        
        p = hparams.get("p", 0.2)
        for in_size, out_size in zip(sizes[:-2], sizes[1:-1]):
            layers.extend([
                nn.Linear(in_size, out_size),
                nn.BatchNorm1d(out_size),
                nn.ReLU(),
                nn.Dropout(p),
            ])
        
        layers.append(nn.Linear(sizes[-2], sizes[-1]))

        self.encoder = encoder
        self.classifier = nn.Sequential(*layers)

        self.set_optimizer()

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, loss_func):
        self.classifier.train()

        self.optimizer.zero_grad()
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        prediction = self(images)
        loss = loss_func(prediction, labels)

        loss.backward()

        self.optimizer.step()
        return loss

    def validation_step(self, batch, loss_func):
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        prediction = self(images)

        loss = loss_func(prediction, labels)
        return loss

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.hparams.get("learning_rate", 1e-3))

    def getAcc(self, loader=None):
        
        assert loader is not None, "Please provide a dataloader for accuracy evaluation"

        self.eval()
        self = self.to(self.device)
            
        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.device)
            score = self.forward(X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc
