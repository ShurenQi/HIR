close all;
clear all;
clc;
addpath(genpath(pwd));
dbstop if error
warning('off'); 
%% parameters 
param = struct();
param.type_feat = 10; % type of feature as following:
%    '1.ZM';'2.PZM';'3.OFMM';'4.CHFM';'5.PJFM';'6.JFM';   % Classical Jacobi polynomial based moments
%    '7.RHFM';'8.EFM';'9.PCET';'10.PCT';'11.PST';         % Classical Harmonic function based moments
%    '12.BFM';                                            % Classical Eigenfunction based moments
%    '13.FJFM';                                           % Fractional-order Jacobi polynomial based moments
%    '14.GRHFM';'15.GPCET';'16.GPCT';'17.GPST'            % Fractional-order Harmonic function based moments
% param.XNM = [0,0;0,1;1,0;0,2;1,1;2,0]; % K = 2 all the n and m for the moments
% param.XNM = [0,0;0,1;1,0;0,2;1,1;2,0;3,0;2,1;1,2;0,3]; %  K = 3 all the n and m for the moments
param.XNM = [0,0;0,1;1,0;0,2;1,1;2,0;3,0;2,1;1,2;0,3;4,0;3,1;2,2;1,3;0,4]; %  K = 4 all the n and m for the moments
param.alpha = 1; % parameter for fractional-order moments
param.p = 2; param.q = 2; % parameters for JFM and FJFM
param.scales = 8; % the scale
param.numofscale = size(param.scales,2); % number of scales
%% load input
img = imread('text.jpg');
img = imresize(img,[512,512],'bicubic');
%% proposed techique
[ord,featcell] = HI(img,param);
%% visualization
K=max(max(param.XNM));
h=figure; set(h,'position',[0 200 3000 800]);
L=size(find(ord==K),2);
idx = 1;
ha = tight_subplot(K+1,L,[0 0],[0 0],[0 0]);
for x = 0:K
    for y=1:size(find(ord==x),2)
        axes(ha(x*L+y));
        imshow(featcell{idx},[]);
        idx=idx+1;
    end
    for z=y+1:L
        axes(ha(x*L+z));
        imshow(ones(size(featcell{idx})),[]);
    end
end




