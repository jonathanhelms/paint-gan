P-GAN
ML program to create a GAN model based on a set of given images. Images were collected on various kaggle.com collections: Van Gogh paintings, Wiki-art : Visual art encyclopedia, and Art portraits.


Generative Adversarial Networks (GANs) have been used in recent years as a tool to create
models for both supervised and unsupervised learning, such as the MINST dataset for creating
realistic looking “handwritten” digits. This paper will give a summary of experiments done using
a GAN model to create works of art starting with specific artists (Van Gogh), to specific genres
of art, such as portraits, landscapes, and abstracts. The model was written with help from GitHub
and this document. An updated version was attempted with the added benefit of a
Wasserstein function, but poor results were achieved. All the datasets were collected from
Kaggle, with the Van Gogh works from this reference, the landscapes and abstract images
were from this reference, and the portraits were from this final reference. The way a GAN
works is there are 2 neural networks competing against each other, where the first network is the
discriminator which trains the real images, and the generator creates images. The generator starts
by creating an image of random noise and over time it should get better at tricking the
discriminator into thinking the generated images are real.
