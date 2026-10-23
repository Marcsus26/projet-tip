# The Food-11 classification challenge

This document presents the collective classification challenge on the [Food-11 dataset](https://www.epfl.ch/labs/mmspg/downloads/food-image-datasets/). It defines the problem and data, sets the rules of organization and provides pointers to tools and methods to help you accomplish the challenge.

## 1. Problem statement

This project has adapted the original Food-11 dataset for the requirements of this challenge. It contains 16393 food images grouped in 11 food categories: Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood, Soup, and Vegetable/Fruit.

The images are divided into 2 sets: a train set of 13296 images and test set of 3097 images. In the `data/food-11` folder, images in the train set are organized into directories of respective food classes. Unsurprisingly, class information is not provided for images in the test set.

Images in the train set can be freely distributed into training and validation subsets in your experiments to choose training hyper-parameters, study different network architectures etc.

Your goal is to implement a machine learning workflow, possibly following a transfer learning strategy, to train, validate and analyze the performances of at least two different neural networks on the train set of the Food-11 dataset, and then test the best performing one on the test set of the Food-11 dataset. To perform testing, you will produce a [JSON file](https://en.wikipedia.org/wiki/JSON) with content having the following format:

```
{
    "0": "bread",
    "1": "bread",
    "2": "fried",
    "3": "bread",
    # other test images
}
```

In this JSON file, the name (without extension) of each test image is associated with the lowercase name of the class predicted by your network. The class names in the JSON file must correspond to the names of the directories in the train set. Make sure that the JSON file you produce **follows this format strictly**.

Submissions are ranked by the **Frugality score** proposed by [Evchenko et al.](https://arxiv.org/abs/2111.03731) It is defined as follows:

$$
Frug = P - \frac{w}{1 + \frac{1}{R}}
$$

where $P$ is the prediction score, set to F1 in this challenge; $R$ is the ressource use, set to the number of network parameters in this challenge ; $w$ controls the relative importance of predictive and frugal performances, set to $0.3$ in this challenge. Obviously, the Frugality score offsets predicitive performances by ressource consumption. For example, a network of 3.5 M parameters achieving F1 = 0.80 would have a Frugality score of 0.57, ranking higher than a 11.6 M parameter network achieving F1 = 0.82 which would have a Frugality score of 0.54.

It is **prohibited** to use another distribution of the Food-11 dataset for training.


## 2. Tools

You should use the [Deep Learning Toolbox](https://fr.mathworks.com/help/deeplearning/index.html) of Matlab for this project. In addition to a rich collection of functions and tools to build, train and evaluate networks, it provides library facilities to [load and prepare image data](https://fr.mathworks.com/help/deeplearning/ug/preprocess-images-for-deep-learning.html) for training and inference. Matlab also provides a collection of [pre-trained networks](https://fr.mathworks.com/help/deeplearning/ug/pretrained-convolutional-neural-networks.html) which come handy for tasks of classification and visual recognition on [new image data](https://fr.mathworks.com/help/deeplearning/ug/retrain-neural-network-to-classify-new-images.html) following [transfer learning](https://fr.mathworks.com/help/deeplearning/gs/get-started-with-transfer-learning.html) strategies.
