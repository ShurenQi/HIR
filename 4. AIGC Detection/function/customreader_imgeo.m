function data = customreader_imgeo(filename,imgsize)
onState = warning('off', 'backtrace');
c = onCleanup(@() warning(onState));
data = imread(filename);
RA = [0,90,180,270];
i = randi([1,6],1,1);
if i<=4
    data = imrotate(data,RA(i),'bicubic','crop');
elseif i == 5
    data = flip(data,1);
else
    data = flip(data,2);
end
data = imresize(data,'OutputSize',[imgsize,imgsize]);
if size(data,3) == 1
    data = repmat(data,[1 1 3]);
end
end