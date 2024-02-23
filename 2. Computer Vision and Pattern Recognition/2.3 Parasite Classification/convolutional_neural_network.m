%%
close all;
clear all;
clc;
warning('off');
addpath(genpath(pwd));
dbstop if error
%% Parasite Images
location = fullfile(pwd,'Parasite Data Set');
Imds = imageDatastore(location,'IncludeSubfolders',true, 'LabelSource','foldernames');
rng(10);
Imds = shuffle(Imds);
[trainImds,testImds] = splitEachLabel(Imds,0.1);
countEachLabel(trainImds)
countEachLabel(testImds)
%% Parasite Featuring with CNN
imgsize = 224;
trainImds.ReadFcn = @(x)repmat(imresize(im2gray(imread(x)),'OutputSize',[imgsize,imgsize]),[1,1,3]);
testImds.ReadFcn = @(x)repmat(imresize(im2gray(imread(x)),'OutputSize',[imgsize,imgsize]),[1,1,3]);
reset(trainImds);
reset(testImds);
figure;
for np = 1:20
    subplot(4,5,np)
    im = read(testImds);
    imagesc(im);    
    colormap gray; axis off;
end
%% --------simple CNN imgsize = 224
layers = [
    imageInputLayer([imgsize,imgsize,3])
    convolution2dLayer(7,16)
    batchNormalizationLayer
    reluLayer    
    convolution2dLayer(3,20)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(4)
    fullyConnectedLayer(8)
    softmaxLayer
    classificationLayer];
%% --------alexnet imgsize = 227
% inputnet = alexnet;
% layersTransfer = inputnet.Layers(1:end-3);
% numClasses = numel(categories(trainImds.Labels));
% layers = [
%     layersTransfer
%     fullyConnectedLayer(numClasses)
%     softmaxLayer
%     classificationLayer];
%% --------vgg16 imgsize = 224
% inputnet = vgg16;
% layersTransfer = inputnet.Layers(1:end-3);
% numClasses = numel(categories(trainImds.Labels));
% layers = [
%     layersTransfer
%     fullyConnectedLayer(numClasses)
%     softmaxLayer
%     classificationLayer];
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
%% Parasite Classification with CNN Model
predlabels = classify(net,testImds,'ExecutionEnvironment','gpu');
Accuracy = sum(testImds.Labels == predlabels)./numel(testImds.Labels)*100;
figure;
cchart = confusionchart(testImds.Labels,predlabels);
title(['CNN, accuracy = ',num2str(Accuracy),'%']);
[avgmetrics, permetrics] = multiclass_metrics_special(cchart.NormalizedValues);
clc;
disp(table(Time,...
    'RowNames',{'Time'},'VariableNames',{'Seconds'}));
disp(table([avgmetrics.Accuracy;avgmetrics.Recall;avgmetrics.Precision;avgmetrics.F1_score]*100,...
    'RowNames',{'Accuracy';'Recall';'Precision';'F1'},'VariableNames',{'Avg. metrics'}));
disp(table(permetrics.F1_score*100,...
    'RowNames',categories(cchart.ClassLabels),'VariableNames',{'F1 per class'}));