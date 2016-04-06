function gp = gp_der(gp, varargin)
%GP_SET  Create a Gaussian process model structure. 
%
%  Description
%
% Copyright (c) 2016 Eero Siivola
  
% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

% parse inputs
ip=inputParser;
ip.FunctionName = 'GP_DER';
ip.addRequired('gp',@isstruct);
ip.addOptional('deriv_x_vals', [],  @(x) isreal(x) && all(isfinite(x(:))));
ip.addOptional('deriv_y_vals', [],  @(x) isreal(x) && all(isfinite(x(:)))); % matrix or row vector
ip.addOptional('deriv_i', [],  @(x) isreal(x) && all(isfinite(x(:)))); %  matrix consisting of ones, tells whether or not this partial derivative is known

ip.parse(gp, varargin{:});

deriv_x_vals = ip.Results.deriv_x_vals;

[n,m] =  size(deriv_x_vals);

if(n==0 || m==0)
  print('error, x must be given');
  return;
end


deriv_y_vals = ip.Results.deriv_y_vals;

if(~ (size(deriv_y_vals,2) == m )) % User did not give derivative values
    deriv_y_vals = ones(1,m); % It is assumed that derivative in all directions is positive
end
if(~( size(deriv_y_vals,1) == n))
    deriv_y_vals = repmat(deriv_y_vals(1,:), n, 1);
end

deriv_i = logical(ip.Results.deriv_i);

if(~ (size(deriv_i,2) == m))
    deriv_i = logical(ones(1,m)); % It is assumed that derivative in all directions is positive
end
if(~( size(deriv_i,1) == size(deriv_x_vals,1)))
    deriv_i = repmat(deriv_i(1,:), n, 1);
end
x_unique = [];
y_unique = [];
i_unique = logical([]);
for(i=1:n)
    if(sum(any(deriv_i(i,:)))>0)
        if ~isempty(x_unique)
            [~,index] = ismember(deriv_x_vals(i,:), x_unique,'rows');
        else
            index = 0;
        end
        if(index == 0)
            x_unique = [x_unique; deriv_x_vals(i,:)];
            y_unique = [y_unique; deriv_y_vals(i,:)];
            i_unique = [i_unique; deriv_i(i,:)];
        else
            y_unique(index, deriv_i(i,:)) = deriv_y_vals(i, deriv_i(i,:));
            i_unique(index, (i_unique(index,:) + deriv_i(i,:) )>0) = 1;
        end
    end
end
if(isempty(x_unique))
    print('Error, index set for active derivatives must be nonzero if given');
    return;
end
y_unique(~i_unique)=0;
gp.deriv_x_vals = x_unique;
gp.deriv_y_vals = y_unique;
gp.deriv_i = i_unique;
gp.derivobs = 1;
end
