from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers import RMSprop
import cv2
import matplotlib.image as mpimg
import tensorflow as tf


# path_to_images = 'VanGogh/'
# path_to_images = 'landscape/'
# path_to_images = 'abstract/'
path_to_images = 'ports/'

img_rows = 100
img_cols = 100
channels = 3
img_shape = (img_rows, img_cols, channels)


def build_generator():
    
    noise_shape = (100,)
    
    model = Sequential()
    
    model.add(Dense(256, input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))
    model.summary()
    
    noise = Input(shape=noise_shape)
    img = model(noise)
    
    return Model(noise, img)
    
 
def build_discriminator():
    
    model = Sequential()
    
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    
    img = Input(shape=img_shape)
    validity = model(img)
    
    return Model(img, validity)
    
  
from keras import backend
 
# implementation of wasserstein loss
def wasserstein_loss(y_true, y_pred):
    return backend.mean(y_true * y_pred)
    

from keras.constraints import Constraint
# clip model weights to a given hypercube
class ClipConstraint(Constraint):
    def __init__(self, clip_value):
        self.clip_value = clip_value
 
    # clip model weights to hypercube
    def __call__(self, weights):
        return backend.clip(weights, -self.clip_value, self.clip_value)
 
    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}
        
        
def train(epochs, batch_size=128, save_interval=50):
    X_train = []
    
    for i in range(3000):
        next_img = cv2.imread(path_to_images + '/' + 'ports' + str(int(i)).zfill(5) + '.jpg')
        
        if next_img is None:
            print('Wrong path:', path_to_images + '/' + 'ports' + str(int(i)).zfill(5) + '.jpg')
        else:
            next_process1 = cv2.resize(next_img,(img_rows, img_cols))
            next_process1 = np.array(next_process1)
            next_process = (next_process1.astype(float) - 127.5) / 127.5
            X_train.append(next_process)

    X_train = np.array(X_train)
    
    half_batch = int(batch_size / 2)
    
    
    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        imgs = X_train[idx]
        
        noise = np.random.normal(0, 1, (half_batch, 100))
        
        gen_imgs = generator.predict(noise)
        
        gen_imgs = np.array(gen_imgs)
        
        imgs = np.array(imgs)
        
        ones =np.ones((half_batch, 1))
        zeros =np.zeros((half_batch, 1))
        
        d_loss_real = discriminator.train_on_batch(imgs, ones)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, zeros)
        
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        noise = np.random.normal(0, 1, (batch_size, 100))
        
        valid_y = np.array([1] * batch_size)
        
        g_loss = combined.train_on_batch(noise, valid_y)
        
        print("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
        
        if epoch % save_interval == 0:
            save_imgs(epoch)
            
def save_imgs(epoch):
    r, c = 2, 2
    noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs = generator.predict(noise)
    
    gen_imgs = 0.5 * gen_imgs + 0.5
    
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt])
            axs[i,j].axis('off')
            cnt += 1
            
    fig.savefig("portimages/port_%d.png" % epoch)
    plt.close()
    
 
optimizer = Adam(0.00005, 0.5)

discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])


generator = build_generator()
generator.compile(loss='binary_crossentropy', optimizer=optimizer)

z = Input(shape=(100,))
img = generator(z)

discriminator.trainable = False

valid = discriminator(img)

combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)

train(epochs=15001, batch_size=128, save_interval=1000)

generator.save('generator_modelport4.h5')