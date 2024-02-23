function [BF_x,BF_y] = KM_BF(SZI,K,P_x,P_y)
N = SZI(1);
x       = 0:1:N-1;
BF_x=zeros(K+1,N);
W=zeros(1,N);
W(1,1)=(1-P_x)^N;     %xy=0
for i=1:N-1        %xy=0:1:N-2
    xy=i-1;
    W(i+1)=((N-xy)/(xy+1))*(P_x/(1-P_x))*W(i);
end
if K>=2
    BF_x(1,:)=sqrt(W);
    BF_x(2,:)=((P_x*N-x)/sqrt(P_x*(1-P_x)*N)).*BF_x(1,:);
    for n=3:1:K+1
        order=n-1;
        A = (N*P_x+(order-1)*(1-2*P_x)-x)/sqrt(P_x*(1-P_x)*order*(N-n+2));
        B = sqrt((order-1)*(N-order+2)/(order*(N-order+1)));
        BF_x(n,:)=A.*BF_x(n-1,:)-B.*BF_x(n-2,:);
    end
elseif K==1
    BF_x(1,:)=sqrt(W);
    BF_x(2,:)=((P_x*N-x)/sqrt(P_x*(1-P_x)*N)).*BF_x(1,:);
elseif K==0
    BF_x(1,:)=sqrt(W);
end

M  = SZI(2);
y       = 0:1:M-1;
BF_y=zeros(K+1,M);
W=zeros(1,M);
W(1,1)=(1-P_y)^M;     %xy=0
for i=1:M-1        %xy=0:1:M-2
    xy=i-1;
    W(i+1)=((M-xy)/(xy+1))*(P_y/(1-P_y))*W(i);
end
if K>=2
    BF_y(1,:)=sqrt(W);
    BF_y(2,:)=((P_y*M-y)/sqrt(P_y*(1-P_y)*M)).*BF_y(1,:);
    for n=3:1:K+1
        order=n-1;
        A = (M*P_y+(order-1)*(1-2*P_y)-y)/sqrt(P_y*(1-P_y)*order*(M-n+2));
        B = sqrt((order-1)*(M-order+2)/(order*(M-order+1)));
        BF_y(n,:)=A.*BF_y(n-1,:)-B.*BF_y(n-2,:);
    end
elseif K==1
    BF_y(1,:)=sqrt(W);
    BF_y(2,:)=((P_y*M-y)/sqrt(P_y*(1-P_y)*M)).*BF_y(1,:);
elseif K==0
    BF_y(1,:)=sqrt(W);
end
% X=BF_y*double(img)*BF_x';
end