import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import os
import tqdm


BATCH_SIZE = 100

kwargs = {'num_workers': 1, 'pin_memory': True}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True, **kwargs)

SAVE_DIR = 'drive/Colab/VAE_ABC'
MODEL_FOLDERS = ('results', 'checkpoints')


class BaseVAE(nn.Module):
    def __init__(self, name, base_dir=SAVE_DIR, folders=MODEL_FOLDERS):
        super(BaseVAE, self).__init__()

        self.name = name
        self.base_dir = base_dir
        self.save_dir = f'{self.base_dir}/{self.name}'
        self._init_dirs(folders)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def encode(self, x):
        raise NotImplemented

    def decode(self, z):
        raise NotImplemented

    def save_model(self, epoch):
        torch.save(self.state_dict(), self._save_path(epoch))

    def load_model(self, epoch):
        self.load_state_dict(torch.load(self._save_path(epoch)))

    def _save_path(self, epoch):
        return f'{self.save_dir}/checkpoints/epoch_{epoch:02d}.pth'

    def _init_dirs(self, folders):
        for folder in folders:
            os.makedirs(os.path.join(self.save_dir, folder), exist_ok=True)

        print(os.system(f'ls -laR {self.save_dir}'))


class VAE(BaseVAE):
    def __init__(self, name, base_dir=SAVE_DIR, folders=MODEL_FOLDERS):
        super(VAE, self).__init__(name, base_dir, folders)
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))


class LargerVAE(BaseVAE):
    def __init__(self, name, latent_size=30, base_dir=SAVE_DIR, folders=MODEL_FOLDERS):
        super(LargerVAE, self).__init__(name, base_dir, folders)
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 100)
        self.fc31 = nn.Linear(100, latent_size)
        self.fc32 = nn.Linear(100, latent_size)
        self.fc4 = nn.Linear(latent_size, 100)
        self.fc5 = nn.Linear(100, 400)
        self.fc6 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def decode(self, z):
        h4 = F.relu(self.fc4(z))
        h5 = F.relu(self.fc5(h4))
        return torch.sigmoid(self.fc6(h5))


class ConvVAE(BaseVAE):
    def __init__(self, name, num_filters=8, latent_size=20, base_dir=SAVE_DIR, folders=MODEL_FOLDERS):
        super(ConvVAE, self).__init__(name, base_dir, folders)

        self.num_filters = num_filters
        self.post_conv_size = 7
        self.pre_fc_size = (self.post_conv_size ** 2) * num_filters

        self.conv1 = nn.Conv2d(1, num_filters, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, stride=2, padding=1)
        self.fc_conv_latent_mu = nn.Linear(self.pre_fc_size, latent_size)
        self.fc_conv_latent_logvar = nn.Linear(self.pre_fc_size, latent_size)

        self.fc_latent_conv = nn.Linear(latent_size, self.pre_fc_size)
        self.deconv1 = nn.ConvTranspose2d(num_filters, num_filters, 3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(num_filters, 1, 3, stride=2, padding=1, output_padding=1)

    def encode(self, x):
        x_ = x.view(-1, 1, 28, 28)
        h1 = F.relu(self.conv1(x_))
        h2 = F.relu(self.conv2(h1))
        h2_ = h2.view(-1, self.pre_fc_size)
        return self.fc_conv_latent_mu(h2_), self.fc_conv_latent_logvar(h2_)

    def decode(self, z):
        h3 = F.relu(self.fc_latent_conv(z))
        h3_ = h3.view(-1, self.num_filters, self.post_conv_size, self.post_conv_size)
        h4 = F.relu(self.deconv1(h3_))
        h5 = self.deconv2(h4)
        return torch.sigmoid(h5.view(-1, 784))


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(model, optimizer, epoch, train_dataloader):
    model.train()
    train_loss = 0
    print(f'Starting Train Epoch {epoch}')
    for batch_idx, (data, _) in tqdm.tqdm_notebook(enumerate(train_dataloader), total=len(train_dataloader)):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    #         if batch_idx % 20 == 0:
    #             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #                 epoch, batch_idx * len(data), len(train_loader.dataset),
    #                 100. * batch_idx / len(train_loader),
    #                 loss.item() / len(data)))

    print('\n====> Epoch: {} Average train loss: {:.4f}'.format(
        epoch, train_loss / len(train_dataloader.dataset)))


def test(model, epoch, test_dataloader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_dataloader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data.view(100, 1, 28, 28)[:n],
                                        recon_batch.view(100, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                           f'{model.save_dir}/results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_dataloader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def train_and_test(model, optimizer, num_epochs, train_dataloader,
                   test_dataloader, post_epoch_callback=None, start_epoch=1,
                   latent_size=20):
    for epoch in range(start_epoch, num_epochs + start_epoch):
        train(model, optimizer, epoch, train_dataloader)
        test(model, epoch, test_dataloader)
        model.save_model(epoch)
        with torch.no_grad():
            sample = torch.randn(64, latent_size).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       f'{model.save_dir}/results/sample_' + str(epoch) + '.png')

        if post_epoch_callback is not None:
            post_epoch_callback(epoch)
