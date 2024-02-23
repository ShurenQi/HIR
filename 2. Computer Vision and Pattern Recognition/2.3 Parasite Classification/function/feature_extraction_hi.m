function [F] =feature_extraction_hi(I,param)
if size(I,3) > 1
    I = rgb2gray(I);
end
I = imresize(I,[param.imgnorsize,param.imgnorsize]);
I = padarray(I,[param.imgpadsize,param.imgpadsize],0,'both');
[ord,featcell] = HI(I,param);
F=zeros(size(ord,2),1);
for i = 1:size(ord,2)
    X = featcell{i};
    Y = mean(mean(X));
    F(i,1) = Y;
end
end