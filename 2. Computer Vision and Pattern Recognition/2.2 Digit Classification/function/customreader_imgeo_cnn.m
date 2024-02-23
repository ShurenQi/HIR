function data = customreader_imgeo_cnn(filename,imgsize)
onState = warning('off', 'backtrace');
c = onCleanup(@() warning(onState));
data = imread(filename);
ra = randi([-180,180],1,1);
data = imrotate(data,ra,'bicubic','crop');
txy = randi([-2,2],1,2);
data = imtranslate(data,txy,'bicubic');
data = repmat(imresize(data,[imgsize,imgsize]),[1,1,3]);
end