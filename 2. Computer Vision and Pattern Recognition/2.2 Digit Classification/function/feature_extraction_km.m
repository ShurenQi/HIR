function [F] =feature_extraction_km(I,K,BF,NP,NB)
if size(I,3) > 1
    I = rgb2gray(I);
end
I = imresize(I,[100,100]);
F = zeros(NP,NB);
for i=1:NP
    %% Moments
    M = BF{i,2}*double(I)*BF{i,1}';
    %% Ring Integral
    [IM] = ring_integral(K,NB,abs(M));
    F(i,:) = IM;
end
F = reshape(F',1,[]);
F = F';
end