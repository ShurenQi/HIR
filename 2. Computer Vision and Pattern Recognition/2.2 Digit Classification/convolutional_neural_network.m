%%
close all;
clear all;
clc;
warning('off');
addpath(genpath(pwd));
dbstop if error
%% Digit Images
location = fullfile(pwd,'DigitDataset');
Imds = imageDatastore(location,'IncludeSubfolders',true, 'LabelSource','foldernames');
rng(10);
Imds = shuffle(Imds);
[trainImds,testImds] = splitEachLabel(Imds,0.8);
countEachLabel(trainImds)
countEachLabel(testImds)
%% Digit Featuring with CNN
figure;
for np = 1:20
    subplot(4,5,np)
    im = read(testImds);
    imagesc(im);    
    colormap gray; axis off;
end
imgsize = 224;
trainImds.ReadFcn = @(x)repmat(imresize(imread(x),'OutputSize',[imgsize,imgsize]),[1,1,3]);
testImds.ReadFcn = @(x)repmat(imresize(imread(x),'OutputSize',[imgsize,imgsize]),[1,1,3]);
reset(trainImds);
reset(testImds);
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
    fullyConnectedLayer(10)
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
%% Digit Classification with CNN Model
predlabels = classify(net,testImds,'ExecutionEnvironment','gpu');
accuracy = sum(testImds.Labels == predlabels)./numel(testImds.Labels)*100;
figure;
cchart = confusionchart(testImds.Labels,predlabels);
title(['CNN, accuracy = ',num2str(accuracy),'%']);
[avgmetrics, permetrics] = multiclass_metrics_special(cchart.NormalizedValues);
clc;
disp(table(Time,...
    'RowNames',{'Time'},'VariableNames',{'Seconds'}));
disp(table([avgmetrics.Accuracy;avgmetrics.Recall;avgmetrics.Precision;avgmetrics.F1_score]*100,...
    'RowNames',{'Accuracy';'Recall';'Precision';'F1'},'VariableNames',{'Avg. metrics'}));
disp(table(permetrics.F1_score*100,...
    'RowNames',categories(cchart.ClassLabels),'VariableNames',{'F1 per class'}));
%% Digit Classification under Arbitrary Rotation and Translation Changes
testImds.ReadFcn =  @(x)customreader_imgeo_cnn(x,imgsize);
reset(testImds);
figure;
for np = 1:20
    subplot(4,5,np)
    im = read(testImds);
    imagesc(im);    
    colormap gray; axis off;
end
predlabels = classify(net,testImds,'ExecutionEnvironment','gpu');
accuracy = sum(testImds.Labels == predlabels)./numel(testImds.Labels)*100;
figure;
cchart = confusionchart(testImds.Labels,predlabels);
title(['CNN, accuracy = ',num2str(accuracy),'%']);
[avgmetrics, permetrics] = multiclass_metrics_special(cchart.NormalizedValues);
disp(table([avgmetrics.Accuracy;avgmetrics.Recall;avgmetrics.Precision;avgmetrics.F1_score]*100,...
    'RowNames',{'Accuracy';'Recall';'Precision';'F1'},'VariableNames',{'Avg. metrics'}));
disp(table(permetrics.F1_score*100,...
    'RowNames',categories(cchart.ClassLabels),'VariableNames',{'F1 per class'}));
%% Digit Re-Featuring and Re-Classification with CNN + Augmenter
augmenter = imageDataAugmenter('RandRotation',[-180 180],'RandXTranslation', [-2 2], ...
    'RandYTranslation',[-2 2]);
augimds = augmentedImageDatastore([imgsize,imgsize,3],trainImds,'DataAugmentation',augmenter);
t = tic;
net = trainNetwork(augimds,layers,options);
Time = toc(t);
predlabels = classify(net,testImds,'ExecutionEnvironment','gpu');
accuracy = sum(testImds.Labels == predlabels)./numel(testImds.Labels)*100;
figure;
cchart = confusionchart(testImds.Labels,predlabels);
title(['CNN, accuracy = ',num2str(accuracy),'%']);
[avgmetrics, permetrics] = multiclass_metrics_special(cchart.NormalizedValues);
disp(table(Time,...
    'RowNames',{'Time'},'VariableNames',{'Seconds'}));
disp(table([avgmetrics.Accuracy;avgmetrics.Recall;avgmetrics.Precision;avgmetrics.F1_score]*100,...
    'RowNames',{'Accuracy';'Recall';'Precision';'F1'},'VariableNames',{'Avg. metrics'}));
disp(table(permetrics.F1_score*100,...
    'RowNames',categories(cchart.ClassLabels),'VariableNames',{'F1 per class'}));
%% Digit Classification under Arbitrary Rotation and Translation Changes
testImds.ReadFcn =  @(x)customreader_imgeo_cnn(x,imgsize);
reset(testImds);
figure;
for np = 1:20
    subplot(4,5,np)
    im = read(testImds);
    imagesc(im);    
    colormap gray; axis off;
end
predlabels = classify(net,testImds,'ExecutionEnvironment','gpu');
accuracy = sum(testImds.Labels == predlabels)./numel(testImds.Labels)*100;
figure;
cchart = confusionchart(testImds.Labels,predlabels);
title(['CNN, accuracy = ',num2str(accuracy),'%']);
[avgmetrics, permetrics] = multiclass_metrics_special(cchart.NormalizedValues);
disp(table([avgmetrics.Accuracy;avgmetrics.Recall;avgmetrics.Precision;avgmetrics.F1_score]*100,...
    'RowNames',{'Accuracy';'Recall';'Precision';'F1'},'VariableNames',{'Avg. metrics'}));
disp(table(permetrics.F1_score*100,...
    'RowNames',categories(cchart.ClassLabels),'VariableNames',{'F1 per class'}));