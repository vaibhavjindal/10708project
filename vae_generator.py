import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from VAE import VAE

bs = 100
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

weights_location = 'weights.pt'
vae = VAE(x_dim=784, h_dim1= 512, h_dim2=256, z_dim=2)
vae.load_state_dict(torch.load(weights_location))

def vaeNewSampleGenerator(vae, samples):    
    vae.eval()
    with torch.no_grad():
        new_samples, _, _ = vae(samples.reshape(samples.shape[0], -1))
        return new_samples.reshape(samples.shape)
