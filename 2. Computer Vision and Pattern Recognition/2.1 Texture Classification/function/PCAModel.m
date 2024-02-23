function model = PCAModel(features,M,Labels)
% This function is only to support wavelet image scattering examples in 
% Wavelet Toolbox. It may change or be removed in a future release.
% model = helperPCAModel(features,M,Labels)

% Copyright 2018 MathWorks

% Initialize structure array to hold the affine model
model = struct('Dim',[],'mu',[],'U',[],'Labels',categorical([]),'s',[]);
model.Dim = M;
% Obtain the number of classes
LabelCategories = categories(Labels);
Nclasses = numel(categories(Labels));
for kk = 1:Nclasses
    Class = LabelCategories{kk};
    % Find indices corresponding to each class
    idxClass = Labels == Class;
    % Extract feature vectors for each class
    tmpFeatures = features(:,idxClass);
    % Determine the mean for each class
    model.mu{kk} = mean(tmpFeatures,2);
    [model.U{kk},model.S{kk}] = scatPCA(tmpFeatures);
    if size(model.U{kk},2) > M
        model.U{kk} = model.U{kk}(:,1:M);
        model.S{kk} = model.S{kk}(1:M);
        
    end
    model.Labels(kk) = Class;
end
    
function [u,s,v] = scatPCA(x,M)
	% Calculate the principal components of x along the second dimension.

	if nargin > 1 && M > 0
		% If M is non-zero, calculate the first M principal components.
	    [u,s,v] = svds(x-sig_mean(x),M);
	    s = abs(diag(s)/sqrt(size(x,2)-1)).^2;
	else
		% Otherwise, calculate all the principal components.
        % Each row is an observation, i.e. the number of scattering paths
        % Each column is a class observation
		[u,d] = eig(cov(x'));
		[s,ind] = sort(diag(d),'descend');
		u = u(:,ind);
	end
end
end