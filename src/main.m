% Food-11 classification: load the data, train a CNN, display the
% confusion matrix on a validation split and, optionally, write the
% submission JSON for the unlabeled test set.

clearvars
close all

%% Configuration
% Set to true to predict the test set and write the submission JSON.
generateSubmission = true;

inputSize = [512 512 3];  % Every image is resized to this size.
validationFraction = 0.2;        % Fraction of each class held out for validation.

% Paths are resolved relative to this script, so it runs from any folder.
srcDir = fileparts(mfilename("fullpath"));
if strlength(srcDir) == 0
    srcDir = pwd;
end
projectDir = fileparts(srcDir);
dataRoot = fullfile(projectDir, "data", "food-11");
trainDir = fullfile(dataRoot, "train");
testDir = fullfile(dataRoot, "test");
outDir = fullfile(projectDir, "out");

%% Load the data
% Train images are organized in one folder per class; the folder names
% are the labels.
imds = imageDatastore(trainDir, ...
    IncludeSubfolders=true, LabelSource="foldernames");

% Stratified split: the same fraction of each class goes to validation.
[imdsTrain, imdsVal] = splitEachLabel(imds, 1 - validationFraction, "randomized");
classNames = categories(imdsTrain.Labels);
numClasses = numel(classNames);
fprintf("Training images: %d, validation images: %d, classes: %d\n", ...
    numel(imdsTrain.Files), numel(imdsVal.Files), numClasses);

% Training images are randomly flipped, shifted and scaled.
augmenter = imageDataAugmenter( ...
    RandXReflection=true, ...
    RandXTranslation=[-20 20], ...
    RandYTranslation=[-20 20], ...
    RandScale=[0.9 1.1]);
augTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    DataAugmentation=augmenter, ColorPreprocessing="gray2rgb");
augValidation = augmentedImageDatastore(inputSize, imdsVal, ...
    ColorPreprocessing="gray2rgb");

%% Define the network
% Global average pooling before the classifier keeps the number of
% parameters independent of the input resolution (frugality score).
% layers = [
%     imageInputLayer(inputSize)
% 
%     convolution2dLayer(3, 64, Padding="same", Stride=2)
%     batchNormalizationLayer()
%     reluLayer()
% 
%     convolution2dLayer(3, 64, Padding="same")
%     batchNormalizationLayer()
%     reluLayer()
% 
%     maxPooling2dLayer(2, Stride=2)
% 
%     convolution2dLayer(3, 128, Padding="same")
%     batchNormalizationLayer()
%     reluLayer()
% 
%     convolution2dLayer(3, 128, Padding="same")
%     batchNormalizationLayer()
%     reluLayer()
% 
%     maxPooling2dLayer(2, Stride=2)
% 
%     fullyConnectedLayer(numClasses)
%     softmaxLayer()
%     ];
% 

PretrainedNet = imagePretrainedNetwork("mobilenetv2", NumClasses=numClasses);
analyzeNetwork(PretrainedNet)

%% Hyperparameters
miniBatchSize = 32;
maxEpochs = 5;
initialLearnRate = 1e-3;
sgdmMomentum = 0.9;
L2Regularization = 5e-4;
learnRateDropPeriod = 18;
learnRateDropFactor = 0.1;

%% Train the network
% Validate roughly once per epoch.
validationFrequency = floor(numel(imdsTrain.Files) / miniBatchSize);

options = trainingOptions("adam", ...
    InitialLearnRate=initialLearnRate, ...
    L2Regularization=L2Regularization, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=learnRateDropPeriod, ...
    LearnRateDropFactor=learnRateDropFactor, ...
    MaxEpochs=maxEpochs, ...
    Shuffle="every-epoch", ...
    MiniBatchSize=miniBatchSize, ...
    ValidationData=augValidation, ...
    ValidationFrequency=validationFrequency, ...
    Verbose=false, ...
    Plots="training-progress", ...
    Metrics="accuracy");


net = trainnet(augTrain, PretrainedNet, "crossentropy", options);

%% Evaluate the final network and display the confusion matrix
% Train-set accuracy is measured on the resized images without augmentation.
augTrainEval = augmentedImageDatastore(inputSize, imdsTrain, ...
    ColorPreprocessing="gray2rgb");
trainScores = minibatchpredict(net, augTrainEval, MiniBatchSize=miniBatchSize);
trainLabelsPred = scores2label(trainScores, classNames);
trainAccuracy = mean(trainLabelsPred == imdsTrain.Labels);
fprintf("Train-set accuracy: %.2f%%\n", 100 * trainAccuracy);

validationScores = minibatchpredict(net, augValidation, MiniBatchSize=miniBatchSize);
validationLabelsPrediction = scores2label(validationScores, classNames);
validationLabelsTrue = imdsVal.Labels;

valAccuracy = mean(validationLabelsPrediction == validationLabelsTrue);
fprintf("Validation accuracy: %.2f%%\n", 100 * valAccuracy);

figure
cm = confusionchart(validationLabelsTrue, validationLabelsPrediction);
cm.ColumnSummary = "column-normalized"; % Precision by predicted class.
cm.RowSummary = "row-normalized";       % Recall by true class.
cm.Title = "Food-11 Confusion Matrix (validation)";

% Per-class F1 score and its macro average.
C = confusionmat(validationLabelsTrue, validationLabelsPrediction, Order=classNames);
precision = diag(C) ./ sum(C, 1)';
recall = diag(C) ./ sum(C, 2);
f1PerClass = 2 * precision .* recall ./ (precision + recall);
f1PerClass(isnan(f1PerClass)) = 0;
macroF1 = mean(f1PerClass);
fprintf("Validation macro F1: %.4f\n", macroF1);

% Frugality score from the README: Frug = F1 - w / (1 + 1/R), with R the
% number of parameters in millions and w = 0.3.
numParams = sum(cellfun(@numel, net.Learnables.Value));
frugality = macroF1 - 0.3 / (1 + 1 / (numParams / 1e6));
fprintf("Number of parameters: %d (%.3f M)\n", numParams, numParams / 1e6);
fprintf("Estimated frugality score: %.4f\n", frugality);

%% Predict the test set and write the submission JSON
if generateSubmission
    imdsTest = imageDatastore(testDir);
    augTest = augmentedImageDatastore(inputSize, imdsTest, ...
        ColorPreprocessing="gray2rgb");

    testScores = minibatchpredict(net, augTest, MiniBatchSize=miniBatchSize);
    testLabelsPred = scores2label(testScores, classNames);

    % Keys are the image names without extension, in numeric order.
    [~, imageNames] = cellfun(@fileparts, imdsTest.Files, UniformOutput=false);
    [~, order] = sort(str2double(imageNames));
    imageNames = string(imageNames(order));
    predictedClasses = lower(string(testLabelsPred(order)));

    entries = compose("    ""%s"": ""%s""", imageNames, predictedClasses);
    jsonText = "{" + newline + strjoin(entries, "," + newline) + newline + "}";

    if ~isfolder(outDir)
        mkdir(outDir);
    end
    timestamp = string(datetime("now", Format="yyyy-MM-dd_HH-mm-ss"));
    submissionFile = fullfile(outDir, "output_" + timestamp + ".json");

    fileID = fopen(submissionFile, "w");
    fprintf(fileID, "%s\n", jsonText);
    fclose(fileID);
    fprintf("Wrote %d test predictions to %s\n", numel(imageNames), submissionFile);
else
    fprintf("Submission JSON not generated (generateSubmission = false).\n");
end
