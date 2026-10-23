## 19/10/2026 - 15:24

### Code
```matlab
%% Define the network
% Global average pooling before the classifier keeps the number of
% parameters independent of the input resolution (frugality score).
layers = [
    imageInputLayer(inputSize)

    convolution2dLayer(3, 64, Padding="same")
    batchNormalizationLayer()
    reluLayer()

    convolution2dLayer(3, 64, Padding="same")
    batchNormalizationLayer()
    reluLayer()

    maxPooling2dLayer(2, Stride=2)

    convolution2dLayer(3, 128, Padding="same")
    batchNormalizationLayer()
    reluLayer()

    convolution2dLayer(3, 128, Padding="same")
    batchNormalizationLayer()
    reluLayer()

    maxPooling2dLayer(2, Stride=2)

    convolution2dLayer(3, 256, Padding="same")
    batchNormalizationLayer()
    reluLayer()

    convolution2dLayer(3, 256, Padding="same")
    batchNormalizationLayer()
    reluLayer()

    maxPooling2dLayer(2, Stride=2)

    fullyConnectedLayer(256)
    reluLayer()
    dropoutLayer(0.5)

    fullyConnectedLayer(numClasses)
    softmaxLayer()
    ];

analyzeNetwork(layers)

%% Hyperparameters
miniBatchSize = 32;
maxEpochs = 30;
initialLearnRate = 1e-3;
sgdmMomentum = 0.9;
L2Regularization = 5e-4;
learnRateDropPeriod = 18;
learnRateDropFactor = 0.1;
```

### Results
> Train-set accuracy: 25.34%<br>
> Validation accuracy: 23.35%<br>
> Validation macro F1: 0.0912<br>
> Number of parameters: 52530507 (52.531 M)<br>
> Estimated frugality score: -0.2032

### Comments
Initial network. This is a network that was trained on `CIFAR10` dataset and obtained a 90% accuracy. However, on `food-11` it gets ~25% accuracy. This descrease in accuracy is probably due to the following issues:

1. CIFAR images are 32.32 while food-11 images are treated as 224x224. Six 3×3 convs with three pools give each output feature a view of about 36×36 pixels. Therefore, on CIFAR that covered the whole image. On a 224×224 food photo it's about 16% of the width, so no single feature ever sees a whole dish.

2. After three pools the feature map is still 28×28×256, about 200k values, which is passed through a `fullyConnectedLayer(256)` that has too many more weights. That's about 5,000 weights per training image

3. The training set has far fewer images (10.6k compared to CIFAR's 50k) and images are less equally distributed, ranging for example, from 376 rice images up to 2000 dessert images

---