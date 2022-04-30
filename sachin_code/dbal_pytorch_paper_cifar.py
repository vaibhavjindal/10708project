PATH = '/home/sachingo/sankalp'
import warnings
warnings.filterwarnings("ignore")

import os
os.makedirs(f'{PATH}/perf_lists_cifar', exist_ok=True)

#!pip install -U skorch
#!pip install modAL

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10
from skorch import NeuralNetClassifier
from modAL.models import ActiveLearner
from torchvision import transforms
# from VAE import VAE

"""### architecture of the network we will be using

We will use the architecture described in the paper.
"""

input_dim, input_height, input_width = 3, 28, 28
class lenet(nn.Module):  
    def __init__(self):
        super(lenet, self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.input_dim = input_dim
        self.class_num = 10

        self.conv1 = nn.Conv2d(self.input_dim, 6, (5, 5), padding=2)
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.class_num)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

"""### read training data"""

cifar10_train = CIFAR10(PATH, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(28, 28))]))
cifar10_test  = CIFAR10(PATH, train=False,download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(28, 28))]))
traindataloader = DataLoader(cifar10_train, shuffle=True, batch_size=60000)
testdataloader  = DataLoader(cifar10_test, shuffle=True, batch_size=10000)
X_train, y_train = next(iter(traindataloader))
X_test , y_test  = next(iter(testdataloader))
X_train, y_train = X_train.detach().cpu().numpy(), y_train.detach().cpu().numpy()
X_test, y_test = X_test.detach().cpu().numpy(), y_test.detach().cpu().numpy()

X_train.shape, X_test.shape

"""### preprocessing"""

# X_train = X_train.reshape(60000, 1, 28, 28)
# X_test = X_test.reshape(10000, 1, 28, 28)

"""### initial labelled data
We initialize the labelled set with 100 balanced randomly sampled examples
"""

initial_idx = np.array([],dtype=int)
for i in range(10):
    idx = np.random.choice(np.where(y_train==i)[0], size=10, replace=False)
    initial_idx = np.concatenate((initial_idx, idx))

X_initial = X_train[initial_idx]
y_initial = y_train[initial_idx]

"""### initial unlabelled pool"""

# X_pool = np.delete(X_train, initial_idx, axis=0)
# y_pool = np.delete(y_train, initial_idx, axis=0)
X_pool = np.copy(X_train)
y_pool = np.copy(y_train)

"""## Query Strategies

### Uniform
All the acquisition function we will use will be compared to the uniform acquisition function $\mathbb{U}_{[0,1]}$ which will be our baseline that we would like to beat.
"""

def uniform(learner, X, n_instances=1):
    query_idx = np.random.choice(range(len(X)), size=n_instances, replace=False)
    return query_idx, X[query_idx]

"""### Entropy
Our first acquisition function is the entropy:
$$ \mathbb{H} = - \sum_{c} p_c \log(p_c)$$
where $p_c$ is the probability predicted for class c. This is approximated by:
\begin{align}
p_c &= \frac{1}{T} \sum_t p_{c}^{(t)} 
\end{align}
where $p_{c}^{t}$ is the probability predicted for class c at the t th feedforward pass.
"""

def max_entropy(learner, X, n_instances=1, T=100):
    random_subset = np.random.choice(range(len(X)), size=2000, replace=False)
    with torch.no_grad():
        outputs = np.stack([torch.softmax(learner.estimator.forward(X[random_subset], training=True),dim=-1).cpu().numpy()
                            for t in range(20)])
    pc = outputs.mean(axis=0)
    acquisition = (-pc*np.log(pc + 1e-10)).sum(axis=-1)
    idx = (-acquisition).argsort()[:n_instances]
    query_idx = random_subset[idx]
    return query_idx, X[query_idx]

def bald(learner, X, n_instances=1, T=100):
    random_subset = np.random.choice(range(len(X)), size=2000, replace=False)
    with torch.no_grad():
        outputs = np.stack([torch.softmax(learner.estimator.forward(X[random_subset], training=True),dim=-1).cpu().numpy()
                            for t in range(20)])
    pc = outputs.mean(axis=0)
    H   = (-pc*np.log(pc + 1e-10)).sum(axis=-1)
    E_H = - np.mean(np.sum(outputs * np.log(outputs + 1e-10), axis=-1), axis=0)  # [batch size]
    acquisition = H - E_H
    idx = (-acquisition).argsort()[:n_instances]
    query_idx = random_subset[idx]
    return query_idx, X[query_idx]

def save_list(input_list, name):
    with open(f'{PATH}/perf_lists_cifar/' + name, 'w') as f:
        for val in input_list:
            f.write("%s\n" % val)

"""### Active Learning Procedure"""

def active_learning_procedure(query_strategy,
                              X_test,
                              y_test,
                              X_pool,
                              y_pool,
                              X_initial,
                              y_initial,
                              estimator,
                              n_queries=150,
                              n_instances=100):
    learner = ActiveLearner(estimator=estimator,
                            X_training=X_initial,
                            y_training=y_initial,
                            query_strategy=query_strategy,
                           )
    perf_hist = [learner.score(X_test, y_test)]
    X_rolling, y_rolling = np.copy(X_initial), np.copy(y_initial)
    for index in range(n_queries):
        query_idx, query_instance = learner.query(X_pool, n_instances)
#         learner.teach(X_pool[query_idx], y_pool[query_idx])
        X_rolling, y_rolling = np.concatenate((X_rolling, X_pool[query_idx]), axis=0), np.concatenate((y_rolling, y_pool[query_idx]))
        learner.fit(X_rolling, y_rolling)
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx, axis=0)
        model_accuracy = learner.score(X_test, y_test)
        print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))
        perf_hist.append(model_accuracy)
    return perf_hist



device = "cuda" if torch.cuda.is_available() else "cpu"
estimator = NeuralNetClassifier(lenet,
                                max_epochs=50,
                                batch_size=100,
                                lr=1.0,
                                optimizer=torch.optim.Adadelta,
                                optimizer__rho=0.9,
                                optimizer__eps=1e-6,
                                criterion=torch.nn.CrossEntropyLoss,
                                train_split=None,
                                verbose=0,
                                device=device)
uniform_perf_hist = active_learning_procedure(uniform,
                                              X_test,
                                              y_test,
                                              X_pool,
                                              y_pool,
                                              X_initial,
                                              y_initial,
                                              estimator,
                                             n_instances=100)
save_list(uniform_perf_hist, "uniform_perf_hist")

device = "cuda" if torch.cuda.is_available() else "cpu"
estimator = NeuralNetClassifier(lenet,
                                max_epochs=50,
                                batch_size=100,
                                lr=1.0,
                                optimizer=torch.optim.Adadelta,
                                optimizer__rho=0.9,
                                optimizer__eps=1e-6,
                                criterion=torch.nn.CrossEntropyLoss,
                                train_split=None,
                                verbose=0,
                                device=device)
bald_perf_hist = active_learning_procedure(bald,
                                              X_test,
                                              y_test,
                                              X_pool,
                                              y_pool,
                                              X_initial,
                                              y_initial,
                                              estimator,
                                             n_instances=100)
save_list(bald_perf_hist, "bald_perf_hist")

device = "cuda" if torch.cuda.is_available() else "cpu"
estimator = NeuralNetClassifier(lenet,
                                max_epochs=50,
                                batch_size=100,
                                lr=1.0,
                                optimizer=torch.optim.Adadelta,
                                optimizer__rho=0.9,
                                optimizer__eps=1e-6,
                                criterion=torch.nn.CrossEntropyLoss,
                                train_split=None,
                                verbose=0,
                                device=device)
entropy_perf_hist = active_learning_procedure(max_entropy,
                                              X_test,
                                              y_test,
                                              X_pool,
                                              y_pool,
                                              X_initial,
                                              y_initial,
                                              estimator,
                                             n_instances=100)
save_list(entropy_perf_hist, "entropy_perf_hist")
