%%
close all;
clear all;
clc;
warning('off');
addpath(genpath(pwd));
dbstop if error
%% Parameters
imgsize = 512; 
K = 500;
P = [0.5,0.5;0.42,0.58;0.58,0.42];
NP = size(P,1);
NB = 75;
BF = cell(NP,2);
SZI = [imgsize,imgsize];
for i=1:NP
    [BF_x,BF_y]=KM_BF(SZI,K,P(i,1),P(i,2));
    BF{i,1}=BF_x; BF{i,2}=BF_y;
end
%% Real and Fake Images
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
%% Image Featuring with KM
figure;
for np = 1:20
    subplot(4,5,np)
    im = read(testImds);
    imagesc(im);    
    colormap gray; axis off;
end
t = tic;
Ttrain = tall(trainImds);
Ttest = tall(testImds);
trainfeatures = cellfun(@(x)feature_extraction_km(x,K,BF,NP,NB),Ttrain,'Uni',0);
testfeatures = cellfun(@(x)feature_extraction_km(x,K,BF,NP,NB),Ttest,'Uni',0);
Trainf = gather(trainfeatures);
trainfeatures = cat(2,Trainf{:});
Testf = gather(testfeatures);
testfeatures = cat(2,Testf{:});
Time = toc(t);
%% Image Classification with NN Model
NNmodel = fitcnet(trainfeatures',trainImds.Labels);
predlabels = predict(NNmodel,testfeatures');
Accuracy = sum(testImds.Labels == predlabels)./numel(testImds.Labels)*100;
figure;
cchart = confusionchart(testImds.Labels,predlabels);
title(['KM, NN, accuracy = ',num2str(Accuracy),'%']);
[metrics, ~] = multiclass_metrics_special(cchart.NormalizedValues);
disp(table(Time,...
    'RowNames',{'Time'},'VariableNames',{'Seconds'}));
disp(table([metrics.Accuracy;metrics.Recall;metrics.Precision;metrics.F1_score]*100,...
    'RowNames',{'Accuracy';'Recall';'Precision';'F1'},'VariableNames',{'Metrics'}));
%% Image Classification with SVM Model
SVMmodel = fitcsvm(trainfeatures',trainImds.Labels,'OptimizeHyperparameters','all','HyperparameterOptimizationOptions',struct('UseParallel',true));
predlabels = predict(SVMmodel,testfeatures');
Accuracy = sum(testImds.Labels == predlabels)./numel(testImds.Labels)*100;
figure;
cchart = confusionchart(testImds.Labels,predlabels);
title(['KM, SVM, accuracy = ',num2str(Accuracy),'%']);
[metrics, ~] = multiclass_metrics_special(cchart.NormalizedValues);
disp(table(Time,...
    'RowNames',{'Time'},'VariableNames',{'Seconds'}));
disp(table([metrics.Accuracy;metrics.Recall;metrics.Precision;metrics.F1_score]*100,...
    'RowNames',{'Accuracy';'Recall';'Precision';'F1'},'VariableNames',{'Metrics'}));
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
Ttest = tall(testImds);
testfeatures = cellfun(@(x)feature_extraction_km(x,K,BF,NP,NB),Ttest,'Uni',0);
Testf = gather(testfeatures);
testfeatures = cat(2,Testf{:});
predlabels = predict(SVMmodel,testfeatures');
Accuracy = sum(testImds.Labels == predlabels)./numel(testImds.Labels)*100;
figure;
cchart = confusionchart(testImds.Labels,predlabels);
title(['KM, SVM, accuracy = ',num2str(Accuracy),'%']);
[metrics, ~] = multiclass_metrics_special(cchart.NormalizedValues);
disp(table(Time,...
    'RowNames',{'Time'},'VariableNames',{'Seconds'}));
disp(table([metrics.Accuracy;metrics.Recall;metrics.Precision;metrics.F1_score]*100,...
    'RowNames',{'Accuracy';'Recall';'Precision';'F1'},'VariableNames',{'Metrics'}));