%%
% close all;
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
             6,0;5,1;4,2;3,3;2,4;1,5;0,6]; %  K = 6 all the n and m for the moments
param.alpha = 1; % parameter for fractional-order moments
param.p = 2; param.q = 2; % parameters for JFM and FJFM
param.scales = 10; % the scale
param.numofscale = size(param.scales,2); % number of scales
param.imgnorsize = 40; % image normaling size
param.imgpadsize = 12; % image pading size
%% Digit Images
location = fullfile(pwd,'DigitDataset');
Imds = imageDatastore(location,'IncludeSubfolders',true, 'LabelSource','foldernames');
rng(10);
Imds = shuffle(Imds);
[trainImds,testImds] = splitEachLabel(Imds,0.8);
countEachLabel(trainImds)
countEachLabel(testImds)
%% Digit Featuring with Hierarchical Invariants
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
Time = toc(t);
%% Digit Classification with PCA Model
model = PCAModel(trainfeatures,30,trainImds.Labels);
predlabels = PCAClassifier(testfeatures,model);
accuracy = sum(testImds.Labels == predlabels)./numel(testImds.Labels)*100;
figure;
cchart = confusionchart(testImds.Labels,predlabels);
title(['Hierarchical Invariants, accuracy = ',num2str(accuracy),'%']);
[avgmetrics, permetrics] = multiclass_metrics_special(cchart.NormalizedValues);
clc;
disp(table(Time,...
    'RowNames',{'Time'},'VariableNames',{'Seconds'}));
disp(table([avgmetrics.Accuracy;avgmetrics.Recall;avgmetrics.Precision;avgmetrics.F1_score]*100,...
    'RowNames',{'Accuracy';'Recall';'Precision';'F1'},'VariableNames',{'Avg. metrics'}));
disp(table(permetrics.F1_score*100,...
    'RowNames',categories(cchart.ClassLabels),'VariableNames',{'F1 per class'}));
%% Digit Classification under Arbitrary Rotation and Translation Changes
testImds.ReadFcn = @customreader_imgeo;
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
predlabels = PCAClassifier(testfeatures,model);
accuracy = sum(testImds.Labels == predlabels)./numel(testImds.Labels)*100;
figure;
cchart = confusionchart(testImds.Labels,predlabels);
title(['Hierarchical Invariants, accuracy = ',num2str(accuracy),'%']);
[avgmetrics, permetrics] = multiclass_metrics_special(cchart.NormalizedValues);
disp(table([avgmetrics.Accuracy;avgmetrics.Recall;avgmetrics.Precision;avgmetrics.F1_score]*100,...
    'RowNames',{'Accuracy';'Recall';'Precision';'F1'},'VariableNames',{'Avg. metrics'}));
disp(table(permetrics.F1_score*100,...
    'RowNames',categories(cchart.ClassLabels),'VariableNames',{'F1 per class'}));

