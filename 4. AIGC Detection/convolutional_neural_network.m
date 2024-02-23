%%
close all;
clear all;
clc;
warning('off');
addpath(genpath(pwd));
dbstop if error
%% Real and Fake Images
imgsize = 224;
RealImds = imageDatastore(fullfile(pwd,'image\Real'),'IncludeSubFolders',true,'LabelSource','foldernames');
FakeImds = imageDatastore(fullfile(pwd,'image\ALLIN'),'IncludeSubFolders',true,'LabelSource','foldernames');
rng(100)
RealImds = shuffle(RealImds);
FakeImds = shuffle(FakeImds);
numofimages = 6000;  % for ALLIN Fake set
% numofimages = length(FakeImds.Files);  % for reset Fake sets
RealImds = subset(RealImds,1:numofimages);
FakeImds = subset(FakeImds,1:numofimages);
Imds = imageDatastore(cat(1, RealImds.Files,FakeImds.Files));
Imds.Labels = cat(1, RealImds.Labels, FakeImds.Labels);
[trainImds,testImds] = splitEachLabel(Imds,0.1);
countEachLabel(trainImds)
countEachLabel(testImds)
%% Image Featuring with CNN
trainImds.ReadFcn = @(x)customreader(x,imgsize);
testImds.ReadFcn = @(x)customreader(x,imgsize);
reset(trainImds);
reset(testImds);
figure;
for np = 1:20
    subplot(4,5,np)
    im = read(testImds);
    imagesc(im);    
    colormap gray; axis off;
end
%% --------simple CNN
layers = [
    imageInputLayer([imgsize,imgsize,3])
    convolution2dLayer(7,16)
    batchNormalizationLayer
    reluLayer    
    convolution2dLayer(3,20)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(4)
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
%% --------alexnet
% inputnet = alexnet;
% layersTransfer = inputnet.Layers(1:end-3);
% numClasses = numel(categories(trainImds.Labels));
% layers = [
%     layersTransfer
%     fullyConnectedLayer(numClasses)
%     softmaxLayer
%     classificationLayer];
%% --------vgg16
% inputnet = vgg16;
% layersTransfer = inputnet.Layers(1:end-3);
% numClasses = numel(categories(trainImds.Labels));
% layers = [
%     layersTransfer
%     fullyConnectedLayer(numClasses)
%     softmaxLayer
%     classificationLayer];
%% --------googlenet
% layers = layerGraph(googlenet);
% numClasses = numel(categories(trainImds.Labels));
% layers = replaceLayer(layers,'loss3-classifier',fullyConnectedLayer(numClasses));
% layers = replaceLayer(layers,'prob',softmaxLayer);
% layers = replaceLayer(layers,'output',classificationLayer);
%% --------resnet50
% layers = layerGraph(resnet50);
% numClasses = numel(categories(trainImds.Labels));
% layers = replaceLayer(layers,'fc1000',fullyConnectedLayer(numClasses));
% layers = replaceLayer(layers,'fc1000_softmax',softmaxLayer);
% layers = replaceLayer(layers,'ClassificationLayer_fc1000',classificationLayer);
%% --------densenet201
% layers = layerGraph(densenet201);
% numClasses = numel(categories(trainImds.Labels));
% layers = replaceLayer(layers,'fc1000',fullyConnectedLayer(numClasses));
% layers = replaceLayer(layers,'fc1000_softmax',softmaxLayer);
% layers = replaceLayer(layers,'ClassificationLayer_fc1000',classificationLayer);
%% --------inceptionv3
% layers = layerGraph(inceptionv3);
% numClasses = numel(categories(trainImds.Labels));
% layers = replaceLayer(layers,'predictions',fullyConnectedLayer(numClasses));
% layers = replaceLayer(layers,'predictions_softmax',softmaxLayer);
% layers = replaceLayer(layers,'ClassificationLayer_predictions',classificationLayer);
%% --------mobilenetv2
% layers = layerGraph(mobilenetv2);
% numClasses = numel(categories(trainImds.Labels));
% layers = replaceLayer(layers,'Logits',fullyConnectedLayer(numClasses));
% layers = replaceLayer(layers,'Logits_softmax',softmaxLayer);
% layers = replaceLayer(layers,'ClassificationLayer_Logits',classificationLayer);
%% --------
analyzeNetwork(layers);
options = trainingOptions('sgdm',...
    'InitialLearnRate',0.0001, ...
    'MaxEpochs',20, ...
    'MiniBatchSize',100,...
    'Shuffle','every-epoch',...
    'Plots', 'training-progress',...
    'Verbose',false,...
    'ExecutionEnvironment','gpu');
t = tic;
net = trainNetwork(trainImds,layers,options);
Time = toc(t);
%% Image Classification with CNN Model
predlabels = classify(net,testImds,'ExecutionEnvironment','gpu');
Accuracy = sum(testImds.Labels == predlabels)./numel(testImds.Labels)*100;
figure;
cchart = confusionchart(testImds.Labels,predlabels);
title(['CNN, accuracy = ',num2str(Accuracy),'%']);
[avgmetrics, ~] = multiclass_metrics_special(cchart.NormalizedValues);
clc;
disp(table(Time,...
    'RowNames',{'Time'},'VariableNames',{'Seconds'}));
disp(table([avgmetrics.Accuracy;avgmetrics.Recall;avgmetrics.Precision;avgmetrics.F1_score]*100,...
    'RowNames',{'Accuracy';'Recall';'Precision';'F1'},'VariableNames',{'Avg. metrics'}));
%% Image Classification under Orientation and Flipping Changes
testImds.ReadFcn = @(x)customreader_imgeo(x,imgsize);
reset(testImds);
figure;
for np = 1:20
    subplot(4,5,np)
    im = read(testImds);
    imagesc(im);    
    colormap gray; axis off;
end
predlabels = classify(net,testImds,'ExecutionEnvironment','gpu');
Accuracy = sum(testImds.Labels == predlabels)./numel(testImds.Labels)*100;
figure;
cchart = confusionchart(testImds.Labels,predlabels);
title(['CNN, accuracy = ',num2str(Accuracy),'%']);
[metrics, ~] = multiclass_metrics_special(cchart.NormalizedValues);
disp(table(Time,...
    'RowNames',{'Time'},'VariableNames',{'Seconds'}));
disp(table([metrics.Accuracy;metrics.Recall;metrics.Precision;metrics.F1_score]*100,...
    'RowNames',{'Accuracy';'Recall';'Precision';'F1'},'VariableNames',{'Metrics'}));