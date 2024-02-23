function data = customreader_imgeo(filename)
onState = warning('off', 'backtrace');
c = onCleanup(@() warning(onState));
data = imread(filename);
ra = randi([-180,180],1,1);
data = imrotate(data,ra,'bicubic','crop');
txy = randi([-2,2],1,2);
data = imtranslate(data,txy,'bicubic');
end