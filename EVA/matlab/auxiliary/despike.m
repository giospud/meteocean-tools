function [x,index] = despike(x,X)

j=1;
index = [];

for i=2:(length(x)-1)
    if(x(i)>X || x(i)<=0)

        x(i) = (x(i-1)+x(i+1))/2;
        index(j) = i;
		j = j+1;
              
    end     
end

return