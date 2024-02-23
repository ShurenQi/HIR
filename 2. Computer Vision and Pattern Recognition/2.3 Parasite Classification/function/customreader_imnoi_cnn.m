function data = customreader_imnoi_cnn(filename,imgsize)
onState = warning('off', 'backtrace');
c = onCleanup(@() warning(onState));
data = im2gray(imread(filename));
% data = imnoise(data,'gaussian',0,0.001);
data = imnoise(data,'salt & pepper',0.001);
data = repmat(imresize(data,[imgsize,imgsize]),[1,1,3]);
end