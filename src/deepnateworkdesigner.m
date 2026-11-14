clearvars;

numClasses = 11;

PretrainedNet = imagePretrainedNetwork("shufflenet", NumClasses=numClasses);
deepNetworkDesigner(PretrainedNet)