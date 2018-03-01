function [ Mnew ] = PartitionN( N )
% We want to find all the partitions of N into N numbers
% If N=3 => 3=0+0+3=0+1+2=1+1+1
% If N=0 => 0 - so one later has to be careful

if (N==0) Mnew=0; return
end


M=partitions(N,1:N);
vector0=(N-sum(M'))';

M0=cat(2,vector0,M);
[m ~]=size(M0);

Mnew=zeros(m,N);

for i=1:m
    Mnew(i,:)=repelem(0:N,M0(m-i+1,:));
end

end
