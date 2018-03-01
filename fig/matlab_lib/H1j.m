function [ H1 ] = H1j( N )
% returns a vector of H0 whose jth element is H0(j-1)
% for j=1,2,...,N
H1=zeros(1,N);

%f=SumDerGamma(N);

% f is a function of sum of product of sum
f=FH1j(N);
fnew=flip(f);

vect23=(2/3).^(0:N-1);
g=vect23.*fnew;

for i=1:N
    j=i-1;
    pom=sum(g(i:end)); % sum goes over 0,..,N-1
    H1(i)=(-1)^N * (2/3)^(1-j)/factorial(j)*pom;
end

end