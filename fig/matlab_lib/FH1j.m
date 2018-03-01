function [ SumDerGammaH1j ] = FH1j( N )
% sum_{j_1+...+j_N=k} \prod_{i=1}^t {\sum_{l1+l2=j_t} factorialgamma(l2)}
% k=0,1,2,...,N-1

SumDerGammaH1j=zeros(1,N);
G=factorialgamma(N);

% h(j_t) = \sum_{l1+l2+j_t} dergamma(l2)
% j_t=0,1,2,...,N-1

h=zeros(1,N);
for i=1:N
    j=i-1;
    if (j==0) Part=[0 0];
    elseif (j==1) Part=[0 1; 1 0];
    else
        M=PartitionN(j);
        Mnew=M(:,end-1:end);
        rows=find(sum(Mnew')'==j);
        Part=Mnew(rows,:);
    end
% we have Part full of partitions
% since we sum only the l2 part it is enough to
% concatenate all of the partitions, look at the 
% unique elements and sum over them
    vector=reshape(Part,1,[]);
    vector=unique(vector);
    h(i)=sum(G(vector+1));    
end

for i=1:N
    k=i-1;
    M1=PartitionN(k);
    if (M1==0) M1new=zeros(1,N); n=1;
    else
    [n, ~]=size(M1);
    M1new=cat(2,zeros(n,N-k),M1);
    end
    %
    numb=NumbOfComb( M1new, N );
    %
    sum1=0;
    for ii=1:n
        part=M1new(ii,:);
        sum1=sum1+numb(ii)*prod(h(part+1));
    end
    SumDerGammaH1j(i)=sum1;
end

end


