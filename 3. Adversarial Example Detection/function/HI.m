function [ord,featcell] = HI(img,param)
K = max(max(param.XNM));
idx = 1;
featcell = cell(1,1);
ord = zeros(1,1);
for x = 0:K
    if x == 0
        param.NM = param.XNM(1,:);
        [feat,~] = DIR(img,param);
        featcell(1)=feat;
        ord(1)=x;
    else
        param.NM = param.XNM(idx:idx+x,:);
        inputfeat=img;
        [feat,~] = DIR(inputfeat,param);
        for z=1:size(feat{1},3)
            featcell = [featcell, {feat{1}(:,:,z)}];
            ord =[ord, x];
        end
        for y = idx-x:idx-1
            param.NM = param.XNM(idx:idx+x,:);
            inputfeat=featcell{y};
            [feat,~] = DIR(inputfeat,param);
            for z=1:size(feat{1},3)
                featcell = [featcell, {feat{1}(:,:,z)}];
                ord =[ord, x];
            end
        end
    end
    idx=idx+x+1;
end
end