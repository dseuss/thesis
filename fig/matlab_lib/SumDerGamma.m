function [ vectorf ] = SumDerGamma( N )
% for k=0,1,...,N-1 we compute \sum_{j_1+...+j_N=k}
% factorialgamma(j1)*...*factorialgamma(j_N)
Gamma=factorialgamma(N);
vectorf=zeros(1,N); % vectorf(i) => k=i-1

for i=1:N
    k=i-1;
    M=PartitionN(k);
    if (M==0) Mnew=zeros(1,N); n=1;
    else [n,~]=size(M); Mnew=cat(2,zeros(n,N-k),M);
    end
    vectcomb=NumbOfComb(Mnew,N);
    suma=0;
    for ii=1:n
        part=Mnew(ii,:);
        pom=prod(Gamma(part+1));
        suma=suma+pom*vectcomb(ii);
    end
    vectorf(i)=suma;
end

    
end

