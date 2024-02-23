function [IM] =ring_integral(K,NB,M)
[Y,X]=meshgrid(1:K+1,1:K+1);
Rad = sqrt(X.^2+Y.^2);
LB=(max(Rad(:))-min(Rad(:)))/NB;
IM=zeros(1,NB);
M=M.*(Rad.^2);
for i=1:NB
    minRad=(i-1)*LB; maxRad=(i)*LB;
    PZ=(Rad>minRad)&(Rad<=maxRad);
    temp=M(PZ);
    IM(i)=sum(temp)/sum(PZ(:));
%     IM(i)=sum(temp);
end
% minRad=(60-1)*LB; maxRad=(80)*LB;
% PZ=(Rad>minRad)&(Rad<=maxRad);
% figure;imshow(PZ);
end