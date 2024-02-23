%%
close all;
clear all;
clc;
warning('off');
addpath(genpath(pwd));
dbstop if error
%% Parameters 
K = 80;
P = [0.5,0.5;0.42,0.58;0.58,0.42];
NP = size(P,1);
NB = 75;
BF = cell(NP,2);
SZI = [100,100];
for i=1:NP
    [BF_x,BF_y]=KM_BF(SZI,K,P(i,1),P(i,2));
    BF{i,1}=BF_x; BF{i,2}=BF_y;
end
%% Texture Images
location = fullfile(pwd,'KTH_TIPS');
Imds = imageDatastore(location,'IncludeSubFolders',true,'FileExtensions','.png','LabelSource','foldernames');
rng(100)
Imds = shuffle(Imds);
[trainImds,testImds] = splitEachLabel(Imds,0.8);
countEachLabel(trainImds)
countEachLabel(testImds)
%% Texture Featuring with KM
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
%% Texture Classification with PCA Model
model = PCAModel(trainfeatures,30,trainImds.Labels);
predlabels = PCAClassifier(testfeatures,model);
Accuracy = sum(testImds.Labels == predlabels)./numel(testImds.Labels)*100;
figure;
cchart = confusionchart(testImds.Labels,predlabels);
title(['KM, accuracy = ',num2str(Accuracy),'%']);
[avgmetrics, permetrics] = multiclass_metrics_special(cchart.NormalizedValues);
clc;
disp(table(Time,...
    'RowNames',{'Time'},'VariableNames',{'Seconds'}));
disp(table([avgmetrics.Accuracy;avgmetrics.Recall;avgmetrics.Precision;avgmetrics.F1_score]*100,...
    'RowNames',{'Accuracy';'Recall';'Precision';'F1'},'VariableNames',{'Avg. metrics'}));
disp(table(permetrics.F1_score*100,...
    'RowNames',categories(cchart.ClassLabels),'VariableNames',{'F1 per class'}));
%% Texture Classification under Orientation and Flipping Changes
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
testfeatures = cellfun(@(x)feature_extraction_km(x,K,BF,NP,NB),Ttest,'Uni',0);
Testf = gather(testfeatures);
testfeatures = cat(2,Testf{:});
predlabels = PCAClassifier(testfeatures,model);
accuracy = sum(testImds.Labels == predlabels)./numel(testImds.Labels)*100;
figure;
cchart = confusionchart(testImds.Labels,predlabels);
title(['KM, accuracy = ',num2str(accuracy),'%']);
[avgmetrics, permetrics] = multiclass_metrics_special(cchart.NormalizedValues);
disp(table([avgmetrics.Accuracy;avgmetrics.Recall;avgmetrics.Precision;avgmetrics.F1_score]*100,...
    'RowNames',{'Accuracy';'Recall';'Precision';'F1'},'VariableNames',{'Avg. metrics'}));
disp(table(permetrics.F1_score*100,...
    'RowNames',categories(cchart.ClassLabels),'VariableNames',{'F1 per class'}));
