function [F] =feature_extraction_hi(I,param)
if size(I,3) > 1
    I = rgb2gray(I);
end
I = imresize(I,[param.imgnorsize,param.imgnorsize]);
I = padarray(I,[param.imgpadsize,param.imgpadsize],0,'both');
[ord,featcell] = HI(I,param);
F=zeros(size(ord,2)*param.binsize,1);
for i = 1:size(ord,2)
    X = featcell{i};
    Y = dct2(X)/(size(X,1)*size(X,2));
    Z = ring_integral(size(Y,1)-1,param.binsize,abs(Y));
    F((i-1)*param.binsize+1:(i)*param.binsize) = Z';
end
end