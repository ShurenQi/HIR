function data = customreader_imnoi(filename,imgsize)
onState = warning('off', 'backtrace');
c = onCleanup(@() warning(onState));
data = im2gray(imread(filename));
data = imnoise(data,'gaussian',0,0.001);
% data = imnoise(data,'salt & pepper',0.001);
data = imresize(data,[imgsize,imgsize]);
end