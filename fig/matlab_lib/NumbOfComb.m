function [ numbcomb ] = NumbOfComb( M, k )
% M is a matrix containing all the partitions of a number k
% each row contains one partition

[N p] = size(M); % N different partions of k into p numbers
numbcomb = zeros(1, N);
for i=1:N
    partition=M(i,:);
    uniquenumb = unique(partition);
    l=length(uniquenumb); % number of different elements in a partition
    
    possib=1;
    total=0;
    bigp=p; % partition in total p numbers
    for j=1:l-1
        Nrep=sum((partition-uniquenumb(j))==0); % how many repetitions of the ith number
        %i,Nrep
        possib=possib*nchoosek(bigp,Nrep);
        bigp=bigp-Nrep;
        %
        total=total+Nrep;
    end
       numbcomb(i)=possib; 
       %
       t=sum(partition-uniquenumb(l)==0);
       total=total+t;
       if (total~=p) warning('Not a proper partitioning')
       end
end

end

