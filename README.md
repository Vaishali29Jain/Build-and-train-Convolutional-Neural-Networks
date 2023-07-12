# Build-and-train-Convolutional-Neural-Networks

# Data 
The dataset is a collection of 30,000 images divided into three categories: dogs, 
food, and vehicles. The images are in .jpg file format. There are 3 color channels RGB. Each 
example is a 64x64 image, and there are 10,000 examples for each category that is dogs, food,
and vehicles.

# AlexNet architecture
The AlexNet architecture has 8 layers which are 5 convolutional layers and 3 fully connected layers. 
It takes input as image of size 227 x 227 x 3 and outputs a probability distribution over 3 classes. 
The architecture has ReLU activation function and dropout regularization to prevent overfitting.
The first layer is convolutional layer with filter size 11x11x3 and 64 filters. It is followed 
by ReLU activation function and max-pooling layer with a filter size of 3x3 and a stride 
of 2. The second layer is a convolutional layer with a filter size of 5x5x64 and 192 filters. 
This layer is followed by a ReLU activation function and a max-pooling layer with a 
filter size of 3x3 and a stride of 2.The third, fourth, and fifth layers are convolutional 
layers with filter sizes of 3x3x192, 3x3x384, and 3x3x256, respectively, and each layer is 
followed by a ReLU activation function.
The sixth layer is a max-pooling layer with a filter size of 3x3 and a stride of 2. The 
seventh and eighth layers are fully connected layers with 4096 neurons each, and a ReLU 
activation function is applied after each layer. Finally, the output layer is a fully 
connected layer, which outputs a probability distribution over 3 classes.

# Modified AlexNet
Accuracy is increased and training time is decreased in  the modified AlexNet version by using following methods:
Batch Normalization.
The modified AlexNet includes batch normalization layers after each convolutional layer and fully connected layer, that normalize
input to each layer making CNN more stable and it train faster. 
Dropout: The modified AlexNet includes dropout layers after the first and second fully connected layers to prevent 
overfitting. 
Changing activation function: 
The original AlexNet used ReLU activation function after each layer. In the modified version, the same activation function has been used, 
but in addition, batch normalization is applied before it.
These improvements helped in reducing the overfitting and increasing accuracy of model. 
The training time also decreased due to batch normalization and dropout. The improvements 
lead to more stable and efficient model.

# Optimizing CNN + Data Argumentation

# Data 
The SVHN dataset is a collection of real-world images of house numbers captured 
from Google Street View. Images are in .jpg file format. It contains images with varying sizes 
and resolutions and includes both cropped and uncropped versions. The primary objective of this 
dataset is to accurately identify the digits in house numbers.
The dataset comprises 10 classes, each corresponding to the 10 digits (0-9). 
The main statistics about the entries in the dataset include a training set with 73,257 images, 
ranging in size from 32x32 to 732x1280 pixels, and a test set with 26,032 images, ranging in size 
from 32x32 to 876x1296 pixels. There are 3 color channels RGB.

# Adjustments in CNN
It has 4 convolutional layers with ReLU 
activation and max pooling layers in between and three fully connected layers which has two 
dropout layers, followed by three linear (fully connected) layers with ReLU activation. The input 
to this model is also a 3-channel image, and the output is a vector of size 10 as there are 10 digits 
(0-9) that is the prediction probability for each class

# Data augmentation on SVHN dataset
The data augmentation methods that is tried on the training dataset are as follows:
• transforms.RandomHorizontalFlip(): To randomly flip image horizontally with 
a probability of 0.5.
• transforms.RandomVerticalFlip(): To randomly flip image vertically with probability of 
0.5.
• transforms.RandomRotation(15): To randomly rotate the image by an angle within range 
[-15 degrees to +15 degrees]
• transforms.Resize((32, 32)): To resize input images to 32x32 pixels.
• transforms.RandomCrop(32, padding=4): This randomly crop image to size of 32x32 
pixels with padding of 4 pixels on all the sides.
• transforms.ToTensor(): To convert image to PyTorch tensor for training the model.
• transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)): To normalize the tensor image with 
mean and standard deviation of 0.5 for all the channels

