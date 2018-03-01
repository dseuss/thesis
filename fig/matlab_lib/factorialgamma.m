function [ gammafact ] = factorialgamma( N )
% \Gamma^{(k)}(1)/k! at the (k+1)th entry
%  for k=0,1,2,....,N-1
syms x;
gammafact=zeros(1,N);
g=gamma(x);

gammafact(1)=1; % k=0
for i=1:N-1
    g=diff(g);
    gammafact(i+1)=double(subs(g,x,1))/factorial(i);
end
end

