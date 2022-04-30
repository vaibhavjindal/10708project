
PATH = '/home/sachingo/sankalp'
import warnings
warnings.filterwarnings("ignore")

import os
os.makedirs(f'{PATH}/perf_lists_cifar_pix', exist_ok=True)
# !mkdir perf_lists_cifar_vae

import keras
import numpy as np
import os
import tensorflow.keras as tk
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

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

class TinyPix2Pix():
    def __init__(self, input_shape=(32, 32, 3), model_dir='pix2pixmodels'):
        self.input_shape = input_shape
        self.model_dir = '/home/sachingo/sankalp/weights'

    def define_unet(self):
        # UNet
        inputs = keras.layers.Input(self.input_shape)

        conv1 = keras.layers.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = keras.layers.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = keras.layers.Dropout(0.5)(conv4)
        pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = keras.layers.Dropout(0.5)(conv5)

        up6 = keras.layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(keras.layers.UpSampling2D(size=(2, 2))(drop5))
        merge6 = keras.layers.concatenate([drop4, up6], axis=3)
        conv6 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = keras.layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(keras.layers.UpSampling2D(size=(2, 2))(conv6))
        merge7 = keras.layers.concatenate([conv3, up7], axis=3)
        conv7 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = keras.layers.Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(keras.layers.UpSampling2D(size=(2, 2))(conv7))
        merge8 = keras.layers.concatenate([conv2, up8], axis=3)
        conv8 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = keras.layers.Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(keras.layers.UpSampling2D(size=(2, 2))(conv8))
        merge9 = keras.layers.concatenate([conv1, up9], axis=3)
        conv9 = keras.layers.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = keras.layers.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = keras.layers.Conv2D(3, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv9)

        self.unet = keras.Model(inputs=inputs, outputs=conv9)


    def define_patchgan(self):
        # PatchNet
        source_image = keras.layers.Input(shape=self.input_shape)
        target_image = keras.layers.Input(shape=self.input_shape)

        merged = keras.layers.Concatenate()([source_image, target_image])

        x = keras.layers.Conv2D(64, 3, strides=2)(merged)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.Dropout(0.25)(x)

        x = keras.layers.Conv2D(128, 3)(x)
        x = keras.layers.BatchNormalization(momentum=0.8)(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.Dropout(0.25)(x)

        x = keras.layers.Conv2D(256, 3)(x)
        x = keras.layers.BatchNormalization(momentum=0.8)(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.Dropout(0.25)(x)

        x = keras.layers.Conv2D(512, 3)(x)
        x = keras.layers.BatchNormalization(momentum=0.8)(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.Dropout(0.25)(x)

        x = keras.layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

        self.patchgan = keras.Model(inputs=[source_image, target_image], outputs=x)
        self.patchgan.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy')


    def define_tinypix2pix(self):
        self.define_unet()
        self.define_patchgan()

        self.patchgan.trainable = False

        input_source = keras.layers.Input(shape=self.input_shape)
        unet_output = self.unet(input_source)

        patchgan_output = self.patchgan([input_source, unet_output])

        self.tinypix2pix = keras.Model(input_source, [patchgan_output, unet_output])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.tinypix2pix.compile(loss=['binary_crossentropy', 'mae'], optimizer=optimizer, loss_weights=[1, 100])


    def real_samples(self, dataset, batchsize):
        ix = np.random.randint(0, dataset.shape[0], batchsize)
        x_realA, x_realB = dataset[ix], dataset[ix]
        # 'real' class labels == 1
        y_real = np.ones((batchsize,) + self.patchgan.layers[-1].output_shape[1:])

        return [x_realA, x_realB], y_real


    def fake_samples(self, x_real):
        x_fake = self.unet.predict(x_real)
        # 'fake' class labels == 0
        y_fake = np.zeros((len(x_fake),) + self.patchgan.layers[-1].output_shape[1:])

        return x_fake, y_fake

vae = TinyPix2Pix(input_shape=(32, 32, 3))
os.makedirs(vae.model_dir, exist_ok=True)

vae.define_tinypix2pix()

vae.patchgan = tf.keras.models.load_model(vae.model_dir + '/unet_' + '14999' + '.h5')
vae.patchgan = tf.keras.models.load_model(vae.model_dir + '/patchgan_' + '14999' + '.h5')
vae.tinypix2pix = tf.keras.models.load_model(vae.model_dir + '/tinypix2pix_' + '14999' + '.h5')

def get_new_images(X):
    # print(X.shape)
    X = torch.permute(X, (0,2,3,1))
    # print(X.shape)
    # return
    X = X.numpy()
    X_hat = vae.tinypix2pix.predict(X)[1]
    # print(type(X_hat))
    X_hat = torch.permute(torch.tensor(X_hat), (0,3,1,2))
    return X_hat

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

# cifar10_train = CIFAR10(PATH, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(28, 28))]))
cifar10_test  = CIFAR10(PATH, train=False,download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(28, 28))]))
cifar10_train = CIFAR10(PATH, train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
# cifar10_test  = CIFAR10(PATH, train=False,download=True, transform=transforms.Compose([transforms.ToTensor()]))
traindataloader = DataLoader(cifar10_train, shuffle=True, batch_size=60000)
testdataloader  = DataLoader(cifar10_test, shuffle=True, batch_size=10000)
X_train, y_train = next(iter(traindataloader))
X_test , y_test  = next(iter(testdataloader))
X_train, y_train = X_train.detach().cpu().numpy(), y_train.detach().cpu().numpy()
X_test, y_test = X_test.detach().cpu().numpy(), y_test.detach().cpu().numpy()


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

X_initial = F.interpolate(torch.tensor(X_train[initial_idx]), size=28).numpy()
y_initial = y_train[initial_idx]
print(X_initial.shape)
print(y_initial.shape)

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
#        outputs = np.stack([torch.softmax(learner.estimator.forward(X[random_subset], training=True),dim=-1).cpu().numpy()
#                            for t in range(20)])
         outputs = np.stack([torch.softmax(learner.estimator.forward(F.interpolate(torch.tensor(X[random_subset]), size=28), training=True),dim=-1).cpu().numpy()
                            for t in range(20)])
    pc = outputs.mean(axis=0)
    acquisition = (-pc*np.log(pc + 1e-10)).sum(axis=-1)
    idx = (-acquisition).argsort()[:n_instances]
    query_idx = random_subset[idx]
    return query_idx, X[query_idx]

def bald(learner, X, n_instances=1, T=100):
    random_subset = np.random.choice(range(len(X)), size=2000, replace=False)
    with torch.no_grad():
        # outputs = np.stack([torch.softmax(learner.estimator.forward(X[random_subset], training=True),dim=-1).cpu().numpy()
                            # for t in range(20)])
        outputs = np.stack([torch.softmax(learner.estimator.forward(F.interpolate(torch.tensor(X[random_subset]), size=28), training=True),dim=-1).cpu().numpy()
                            for t in range(20)])
    pc = outputs.mean(axis=0)
    H   = (-pc*np.log(pc + 1e-10)).sum(axis=-1)
    E_H = - np.mean(np.sum(outputs * np.log(outputs + 1e-10), axis=-1), axis=0)  # [batch size]
    acquisition = H - E_H
    idx = (-acquisition).argsort()[:n_instances]
    query_idx = random_subset[idx]
    return query_idx, X[query_idx]

def save_list(input_list, name):
    with open(f'{PATH}/perf_lists_cifar_pix/' + name, 'w') as f:
        for val in input_list:
            f.write("%s\n" % val)

"""### Active Learning Procedure"""

def active_learning_procedure_vae(query_strategy,
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

        new_samples = get_new_images(torch.tensor(X_pool[query_idx]))
        # print(new_samples.shape)
        new_samples = F.interpolate(new_samples, size=28)
        X_pool_query = F.interpolate(torch.tensor(X_pool[query_idx]), size=28).numpy()
        new_samples = new_samples.detach().cpu().numpy()
        new_samples = np.concatenate((new_samples, X_pool_query))
        # new_samples = np.concatenate((new_samples, X_pool[query_idx]))
        new_labels = np.concatenate((y_pool[query_idx], y_pool[query_idx]))

        X_rolling, y_rolling = np.concatenate((X_rolling, new_samples), axis=0), np.concatenate((y_rolling, new_labels))
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
uniform_perf_hist = active_learning_procedure_vae(uniform,
                                              X_test,
                                              y_test,
                                              X_pool,
                                              y_pool,
                                              X_initial,
                                              y_initial,
                                              estimator,
                                             n_instances=100)
save_list(uniform_perf_hist, "uniform_perf_hist")
print("bald")
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
bald_perf_hist = active_learning_procedure_vae(bald,
                                              X_test,
                                              y_test,
                                              X_pool,
                                              y_pool,
                                              X_initial,
                                              y_initial,
                                              estimator,
                                             n_instances=100)
save_list(bald_perf_hist, "bald_perf_hist")
print('last')
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
entropy_perf_hist = active_learning_procedure_vae(max_entropy,
                                              X_test,
                                              y_test,
                                              X_pool,
                                              y_pool,
                                              X_initial,
                                              y_initial,
                                              estimator,
                                             n_instances=100)
save_list(entropy_perf_hist, "entropy_perf_hist")


