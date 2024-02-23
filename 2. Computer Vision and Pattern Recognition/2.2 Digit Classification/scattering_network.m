%%
close all;
clear all;
clc;
warning('off');
addpath(genpath(pwd));
dbstop if error
%% Parameters 
sf = waveletScattering2('ImageSize',[28 28],'InvarianceScale',28, ...
    'NumRotations',[8 8]);
%% Digit Images
location = fullfile(pwd,'DigitDataset');
Imds = imageDatastore(location,'IncludeSubfolders',true, 'LabelSource','foldernames');
rng(10);
Imds = shuffle(Imds);
[trainImds,testImds] = splitEachLabel(Imds,0.8);
countEachLabel(trainImds)
countEachLabel(testImds)
%% Digit Featuring with Wavelet Scattering
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
%% Digit Classification with PCA Model
model = PCAModel(trainfeatures,30,trainImds.Labels);
predlabels = PCAClassifier(testfeatures,model);
accuracy = sum(testImds.Labels == predlabels)./numel(testImds.Labels)*100;
figure;
cchart = confusionchart(testImds.Labels,predlabels);
title(['Wavelet Scattering, accuracy = ',num2str(accuracy),'%']);
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
testfeatures = cellfun(@(x)feature_extraction_scattering(sf,x),Ttest,'UniformOutput',false);
Testf = gather(testfeatures);
testfeatures = cat(2,Testf{:});
predlabels = PCAClassifier(testfeatures,model);
accuracy = sum(testImds.Labels == predlabels)./numel(testImds.Labels)*100;
figure;
cchart = confusionchart(testImds.Labels,predlabels);
title(['Wavelet Scattering, accuracy = ',num2str(accuracy),'%']);
[avgmetrics, permetrics] = multiclass_metrics_special(cchart.NormalizedValues);
disp(table([avgmetrics.Accuracy;avgmetrics.Recall;avgmetrics.Precision;avgmetrics.F1_score]*100,...
    'RowNames',{'Accuracy';'Recall';'Precision';'F1'},'VariableNames',{'Avg. metrics'}));
disp(table(permetrics.F1_score*100,...
    'RowNames',categories(cchart.ClassLabels),'VariableNames',{'F1 per class'}));