function [F] =feature_extraction_dct(I,NB)
if size(I,3) > 1
    I = rgb2gray(I);
end
I = imresize(I,[512,512]);
X = dct2(I);
F = ring_integral(size(X,1)-1,NB,abs(X));
F = F';
end