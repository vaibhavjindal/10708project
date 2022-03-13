import torch
from torch.autograd import Variable

def sample(mu, logvar):
    std = torch.exp(0.5*logvar)
    rand_z_score = torch.randn_like(std)
    return mu + rand_z_score*std

def vaeGanNewSampleGenerator(images): 
	enc  = torch.load('./checkpoints/enc.pt', map_location=torch.device('cpu'))
	dec  = torch.load('./checkpoints/dec.pt', map_location=torch.device('cpu'))
	disc = torch.load('./checkpoints/disc.pt', map_location=torch.device('cpu'))
	images = Variable(images)
	mu, logvar = enc(images)
	z = sample(mu, logvar)
	reconstructions = dec(z)
	reconstructions = reconstructions.reshape(-1, 1, 28, 28)
	return reconstructions
