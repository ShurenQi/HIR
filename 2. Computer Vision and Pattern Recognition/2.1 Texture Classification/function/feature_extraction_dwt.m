function [F] =feature_extraction_dwt(I,NB,NF)
if size(I,3) > 1
    I = rgb2gray(I);
end
I = imresize(I,[100,100]);

F = zeros(1,NF);
[LoD,HiD] = wfilters('haar','d');
[cA,cH,cV,cD] = dwt2(I,LoD,HiD,'mode','symh');

X = dct2(cA);
Y = ring_integral(size(X,1)-1,NB,abs(X));
F(1:NB) = Y;

X = dct2(cH);
Y = ring_integral(size(X,1)-1,NB,abs(X));
F(NB+1:2*NB) = Y;

X = dct2(cV);
Y = ring_integral(size(X,1)-1,NB,abs(X));
F(2*NB+1:3*NB) = Y;

X = dct2(cD);
Y = ring_integral(size(X,1)-1,NB,abs(X));
F(3*NB+1:4*NB) = Y;

F = F';
end