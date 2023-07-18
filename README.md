# Handwritten digit generation using DCGAN

A Pytorch implemetation of DCGAN on the MNIST dataset.<br>
**DCGAN** or **Deep Convolutional GAN** are a special type of GAN architecture where discriminator and generator are both CNNs. They can be used for many generative tasks and give better results compared to vanilla GAN's.<br><br>
Here developed the DCGAN from scratch and trained on MNIST dataset to generate fake handwritten digits.<br>
The project implementation follows the paper on DCGAN


# Dataset
The MNIST dataset of handwritten digits was used for the project and is available publicly
![MNIST-0000000001-2e09631a_09liOmx](https://github.com/arka57/DCGAN/assets/36561428/08cb2854-7cea-45cc-9d08-1239f76dca4e)


# Model Architecture
The GAN architecture is shown in the figure below. As this is DCGAN so both generator and discriminator are CNN's<br><br>
![754471a](https://github.com/arka57/DCGAN/assets/36561428/e9b905bb-a6ba-40aa-8b45-5f4448645251)<br><br>


 **Generator** is a CNN with ConvTranspose operations which generates fake handwritten digits(Dimension:1X64X64) from some random noise<br>
 **Discriminator** is also a CNN with Convolution operations which takes a image as input(Dimension:1X64X64) and classify whether the image it received is a real one or fake generated one.<br>
Generator and Discriminator architecture is in the below figure<br><br>
![dcgan_architecture](https://github.com/arka57/DCGAN/assets/36561428/0599edba-d33f-4f2d-b6c9-ad359e354c6e)


# Hyperparameters
LEARNING_RATE=2e-4<br>
BATCH_SIZE = 128<br>
IMAGE_SIZE = 64<br>
CHANNELS_IMG = 1(MNIST dataset image has only 1 channel)<br>
NOISE_DIM = 100<br>
NUM_EPOCHS = 100<br>
FEATURES_DISC = 64<br>
FEATURES_GEN = 64<br>

# Results
The model was trained for 30 epochs in adversarial manner and then it was abled to generate decent images,some of them are below<br><br>
![test_final](https://github.com/arka57/DCGAN/assets/36561428/755a34da-9682-431f-b746-6c2b83d69ce3)

# References
DCGAN paper:[https://arxiv.org/abs/1511.06434v1]
