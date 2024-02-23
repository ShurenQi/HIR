function features = feature_extraction_scattering(sf,x)
% This function is only to support examples in the Wavelet Toolbox.
% It may change or be removed in a future release.

% Copyright 2018 MathWorks
if size(x,3) > 1
    x = rgb2gray(x);
end
x = imresize(x,[100 100]);
smat = featureMatrix(sf,x);
features = mean(mean(smat,2),3);
end