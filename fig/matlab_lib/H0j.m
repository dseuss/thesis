function [ H0 ] = H0j( N )
% returns a vector of H0 whose jth element is H0(j-1)
% for j=1,2,...,N
H0=zeros(1,N);

f=SumDerGamma(N);
fnew=flip(f);
vect2=2.^(0:N-1);

g=vect2.*fnew;


for i=1:N
    j=i-1;
    pom=sum(g(i:end)); % sum goes over 0,..,N-1
    H0(i)=2^(1-j)/factorial(j)*pom;
end

end