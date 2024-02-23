%%
close all;
clear all;
clc;
warning('off');
addpath(genpath(pwd));
dbstop if error
%% Parameters 
sf = waveletScattering2('ImageSize',[224 224],'InvarianceScale',200, ...
    'NumRotations',[8 8]);
%% Real and Fake Images
RealImds = imageDatastore(fullfile(pwd,'image\Real'),'IncludeSubFolders',true,'LabelSource','foldernames');
FakeImds = imageDatastore(fullfile(pwd,'image\ALLIN'),'IncludeSubFolders',true,'LabelSource','foldernames');
rng(100)
FakeImds = shuffle(FakeImds);
numofimages = 4980;  % for ALLIN Fake set
% numofimages = length(FakeImds.Files);  % for reset Fake sets
FakeImds = subset(FakeImds,1:numofimages);
Imds = imageDatastore(cat(1, RealImds.Files,FakeImds.Files));
Imds.Labels = cat(1, RealImds.Labels, FakeImds.Labels);
[trainImds,testImds] = splitEachLabel(Imds,0.1);
countEachLabel(trainImds)
countEachLabel(testImds)
%% Image Featuring with Wavelet Scattering
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
trainfeatures = cellfun(@(x)feature_extraction_scattering(sf,x),Ttrain,'UniformOutput',false);
testfeatures = cellfun(@(x)feature_extraction_scattering(sf,x),Ttest,'UniformOutput',false);
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
title(['Wavelet Scattering, NN, accuracy = ',num2str(Accuracy),'%']);
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
title(['Wavelet Scattering, SVM, accuracy = ',num2str(Accuracy),'%']);
[metrics, ~] = multiclass_metrics_special(cchart.NormalizedValues);
disp(table(Time,...
    'RowNames',{'Time'},'VariableNames',{'Seconds'}));
disp(table([metrics.Accuracy;metrics.Recall;metrics.Precision;metrics.F1_score]*100,...
    'RowNames',{'Accuracy';'Recall';'Precision';'F1'},'VariableNames',{'Metrics'}));