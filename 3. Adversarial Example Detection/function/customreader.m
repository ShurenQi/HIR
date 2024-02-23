function data = customreader(filename,imgsize)
onState = warning('off', 'backtrace');
c = onCleanup(@() warning(onState));
data = imread(filename);
data = imresize(data,'OutputSize',[imgsize,imgsize]);
if size(data,3) == 1
    data = repmat(data,[1 1 3]);
end
end