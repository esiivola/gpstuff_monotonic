function [C, Cinv] = gp_dcov(gp, x1, x2, predcf)
%GP_COV  Evaluate covariance matrix between two input vectors.
%
%  Description
%    C = GP_DCOV(GP, X1, X2, PREDCF) takes in Gaussian process GP and
%    two matrixes X1 and X2 that contain input vectors to GP.
%    Returns covariance matrix C between the latent values in the training
%    inputs X1 & XV and the test inputs X2. XV denotes the virtual input we
%    have (GP.XV). The covariance is computed between latent vector [f
%    df/dx_1 df/dx_2, ... df/dx_d], evaluated at X1 for f and XV for df/dx
%    respectively, and [f df/dx_1 df/dx_2, ... df/dx_d] evaluated at X2 
%    for both the f and its gradient.
%
% Copyright (c) 2007-2010 Jarno Vanhatalo
% Copyright (c) 2010 Tuomas Nikoskinen
% Copyright (c) 2014 Ville Tolvanen
% Copyright (c) 2016 Eero Siivola

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

% Split the training data for normal latent input and gradient inputs
x12=x1;
x11= gp.deriv_x_vals; %gp.xv;


% Derivative observations
[n,m]=size(x1);
[n4,m4]=size(x2);
ncf=length(gp.cf);

[nt,mt]=size(x11);
if isfield(gp, 'nvi') || isfield(gp, 'nvd')
  isn = gp.deriv_i(:);
else
  is = repmat(1:nt,1, m);
  isd = repmat(1:mt, nt, 1);
  isda = isd(:)';
  isn = nt*(isda-1) + is;
end

for i1=1:ncf
  gpcf = gp.cf{i1};    
  if m==1
    Gset1 = gpcf.fh.ginput4(gpcf, x11, x2);
    Gset2 = gpcf.fh.ginput4(gpcf, x2, x12);
    Kff = gpcf.fh.cov(gpcf, x12, x2);
    Kdd = gpcf.fh.ginput2(gpcf, x11, x2);
    
    Kdf=Gset1{1};
    Kfd=Gset2{1};
    C = [Kff Kfd'; Kdf Kdd{1}];
    
    % Input dimension is >1
  else
    [n,m]=size(x11);
    [n2,m2]=size(x2);
    
    Kff = gpcf.fh.cov(gpcf, x12, x2); %%covariance between training points and test points
    Gset1 = gpcf.fh.ginput4(gpcf, x11,x2); %% gradient of test points with respect to virtual observations
    Gset2 = gpcf.fh.ginput4(gpcf, x2, x12); %% gradient of test points with respect to training points 
    
    %Gather matrices from Gset (d k(x1,x2) /d x1)
    %Kfd=cat(2,Gset1{ii1});
    
    Kdf = cat(1,Gset1{1:m});
    Kdf = Kdf(isn, :);
    
    %Kfd22=cat(2,Gset2{ii1});
    
    Kdf22=cat(1,Gset2{1:m})';
    
    % both x derivatives, same dimension (to diagonal blocks)
    D = gpcf.fh.ginput2(gpcf, x11, x2);
    % both x derivatives, different dimension (non-diagonal blocks)
    Kdf2 = gpcf.fh.ginput3(gpcf, x11 ,x2);
    
    % Now build up Kdd m*n x m*n2 matrix, which contains all the
    % both partial derivative" -matrices
    % Add the diagonal matrices
    Kdd=blkdiag(D{1:m});
    % Add the non-diagonal matrices to Kdd
    ii3=0;
    for j=0:m-2
      for i=1+j:m-1
        ii3=ii3+1;
        Kdd(i*n+1:(i+1)*n,j*n2+1:j*n2+n2) = Kdf2{ii3};
        Kdd(j*n+1:j*n+n,i*n2+1:(i+1)*n2) = Kdf2{ii3};
      end
    end
    if isfield(gp, 'nvd')
      % Collect the correct gradient dimensions,
      Kdd = Kdd(isn, :);
    end
 
    % Gather all the matrices into one final matrix K which is the
    % training covariance matrix
    C = [Kff Kdf22; Kdf Kdd];
     %C = [Kff; Kdf];
  end
  if i1==1
    CC=C;
  else
    CC=CC+C;
  end
end
C=CC;
