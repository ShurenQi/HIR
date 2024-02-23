%%
close all;
clear all;
clc;
warning('off');
addpath(genpath(pwd));
dbstop if error
%% Parameters 
param = struct();
param.type_feat = 10; % type of feature as following:
%    '1.ZM';'2.PZM';'3.OFMM';'4.CHFM';'5.PJFM';'6.JFM';   % Classical Jacobi polynomial based moments
%    '7.RHFM';'8.EFM';'9.PCET';'10.PCT';'11.PST';         % Classical Harmonic function based moments
%    '12.BFM';                                            % Classical Eigenfunction based moments
%    '13.FJFM';                                           % Fractional-order Jacobi polynomial based moments
%    '14.GRHFM';'15.GPCET';'16.GPCT';'17.GPST'            % Fractional-order Harmonic function based moments
param.XNM = [0,0;...
             0,1;1,0;...
             0,2;1,1;2,0;...
             3,0;2,1;1,2;0,3;...
             4,0;3,1;2,2;1,3;0,4;...
             5,0;4,1;3,2;2,3;1,4;0,5;...
             6,0;5,1;4,2;3,3;2,4;1,5;0,6;...
             7,0;6,1;5,2;4,3;3,4;2,5;1,6;0,7]; %  K = 7 all the n and m for the moments
param.alpha = 1; % parameter for fractional-order moments
param.p = 2; param.q = 2; % parameters for JFM and FJFM
param.scales = 6; % the scale
param.numofscale = size(param.scales,2); % number of scales
param.imgnorsize = 512; % image normaling size
param.imgpadsize = 0; % image pading size
param.binsize = 30; % number of bins for DCT band integration
param.featsize = 500; % compressed feature size
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
%% Image Featuring with Hierarchical Invariants
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
trainfeatures = cellfun(@(x)feature_extraction_hi(x,param),Ttrain,'Uni',0);
testfeatures = cellfun(@(x)feature_extraction_hi(x,param),Ttest,'Uni',0);
Trainf = gather(trainfeatures);
trainfeatures = cat(2,Trainf{:});
Testf = gather(testfeatures);
testfeatures = cat(2,Testf{:});
idx = fscchi2(trainfeatures',trainImds.Labels);
shorttrainfeatures = trainfeatures(idx(1:param.featsize),:);
shorttestfeatures = testfeatures(idx(1:param.featsize),:);
clear trainfeatures testfeatures
Time = toc(t);
%% Image Classification with NN Model
NNmodel = fitcnet(shorttrainfeatures',trainImds.Labels);
predlabels = predict(NNmodel,shorttestfeatures');
Accuracy = sum(testImds.Labels == predlabels)./numel(testImds.Labels)*100;
figure;
cchart = confusionchart(testImds.Labels,predlabels);
title(['Hierarchical Invariants, NN, accuracy = ',num2str(Accuracy),'%']);
[metrics, ~] = multiclass_metrics_special(cchart.NormalizedValues);
disp(table(Time,...
    'RowNames',{'Time'},'VariableNames',{'Seconds'}));
disp(table([metrics.Accuracy;metrics.Recall;metrics.Precision;metrics.F1_score]*100,...
    'RowNames',{'Accuracy';'Recall';'Precision';'F1'},'VariableNames',{'Metrics'}));
%% Image Classification with SVM Model
SVMmodel = fitcsvm(shorttrainfeatures',trainImds.Labels,'OptimizeHyperparameters','all','HyperparameterOptimizationOptions',struct('UseParallel',true));
predlabels = predict(SVMmodel,shorttestfeatures');
Accuracy = sum(testImds.Labels == predlabels)./numel(testImds.Labels)*100;
figure;
cchart = confusionchart(testImds.Labels,predlabels);
title(['Hierarchical Invariants, SVM, accuracy = ',num2str(Accuracy),'%']);
[metrics, ~] = multiclass_metrics_special(cchart.NormalizedValues);
disp(table(Time,...
    'RowNames',{'Time'},'VariableNames',{'Seconds'}));
disp(table([metrics.Accuracy;metrics.Recall;metrics.Precision;metrics.F1_score]*100,...
    'RowNames',{'Accuracy';'Recall';'Precision';'F1'},'VariableNames',{'Metrics'}));
%% Image Classification under Orientation and Flipping Changes
testImds.ReadFcn = @(x)customreader_imgeo(x,param.imgnorsize);
reset(testImds);
figure;
for np = 1:20
    subplot(4,5,np)
    im = read(testImds);
    imagesc(im);    
    colormap gray; axis off;
end
Ttest = tall(testImds);
testfeatures = cellfun(@(x)feature_extraction_hi(x,param),Ttest,'Uni',0);
Testf = gather(testfeatures);
testfeatures = cat(2,Testf{:});
shorttestfeatures = testfeatures(idx(1:param.featsize),:);
predlabels = predict(SVMmodel,shorttestfeatures');
Accuracy = sum(testImds.Labels == predlabels)./numel(testImds.Labels)*100;
figure;
cchart = confusionchart(testImds.Labels,predlabels);
title(['Hierarchical Invariants, SVM, accuracy = ',num2str(Accuracy),'%']);
[metrics, ~] = multiclass_metrics_special(cchart.NormalizedValues);
disp(table(Time,...
    'RowNames',{'Time'},'VariableNames',{'Seconds'}));
disp(table([metrics.Accuracy;metrics.Recall;metrics.Precision;metrics.F1_score]*100,...
    'RowNames',{'Accuracy';'Recall';'Precision';'F1'},'VariableNames',{'Metrics'}));
