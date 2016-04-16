function [K, C] = gp_dtrcov(gp, x1, x2, predcf)
%GP_DTRCOV  Evaluate training covariance matrix (gp_cov + noise covariance).
%
%  Description
%    K = GP_DTRCOV(GP, X1, XV, PREDCF) takes in Gaussian process GP and
%    matrix X1 that contains training input vectors to GP with XV that 
%    contains the virtual inputs. Returns the covariance matrix K between
%    elements of latent vector [f df/dx_1 df/dx_2, ..., df/dx_d] where f is
%    evaluated at X1 and df/dx_i at XV.
%
%    [K, C] = GP_DTRCOV(GP, TX, PREDCF) returns also the (noisy)
%    covariance matrix C for observations y, which is sum of K and
%    diagonal term, for example, from Gaussian noise.
%
%  See also
%    GP_SET, GPCF_*
%
% Copyright (c) 2006-2010 Jarno Vanhatalo
% Copyright (c) 2010 Tuomas Nikoskinen
% Copyright (c) 2014 Ville Tolvanen
% Copyright (c) 2016 Eero Siivola

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.
% IT IS ASSUMED THAT x2 == XV
if (isfield(gp,'derivobs') && gp.derivobs)
  ncf=length(gp.cf);
  [n,m]=size(x2);
  if sum(size(x2) == size(gp.deriv_x_vals))<2
    is = repmat(1:n,1, m);
    isd = repmat(1:m, n, 1);
    isda = isd(:)';
    isn = n*(isda-1) + is;
  else
    isn = gp.deriv_i(:);
  end
  K=zeros(length(x1)+sum(logical(isn)));
  % Loop over covariance functions
  for i=1:ncf
    % Derivative observations
    gpcf = gp.cf{i};           
    if m==1
      Kff = gpcf.fh.trcov(gpcf, x1);
      Gset = gpcf.fh.ginput4(gpcf, x2,x1);
      D = gpcf.fh.ginput2(gpcf, x2, x2);
      Kdf=Gset{1};
      Kdd=D{1};
      
      % Add all the matrices into a one K matrix
      K = K+[Kff Kdf'; Kdf Kdd];
      [a b] = size(K);
      
      % MULTIDIMENSIONAL input dim >1
    else
      Kff = gpcf.fh.trcov(gpcf, x1);
      if ~isequal(x2,x1)
        G = gpcf.fh.ginput4(gpcf, x2,x1);
      else
        G = gpcf.fh.ginput4(gpcf, x2);
      end
      D= gpcf.fh.ginput2(gpcf, x2, x2);
      Kdf2 = gpcf.fh.ginput3(gpcf, x2 ,x2);
      
      Kfd = cat(1,G{1:m});
      Kfd = Kfd(isn,:);
%       Kfd=[G{1:m}];


      % Now build up Kdd m*n x m*n matrix, which contains all the
      % both partial derivative" -matrices
      Kdd=blkdiag(D{1:m});
      
      % Gather non-diagonal matrices to Kddnodi
      if m==2
        Kddnodi=[zeros(n,n) Kdf2{1};Kdf2{1}' zeros(n,n)];
      else
        t1=1;
        Kddnodi=zeros(m*n,m*n);
        for i=1:m-1
          aa=zeros(1,m);
          t2=t1+m-2-(i-1);
          aa(1,i)=1;
          k=kron(aa,cat(1,zeros((i)*n,n),Kdf2{t1:t2}));
          %k(1:n*m,:)=[];
          k=k+k';
          Kddnodi = Kddnodi + k;
          t1=t2+1;
        end
      end
      % Sum the diag + no diag matrices
      Kdd=Kdd+Kddnodi;
      
      Kdd = Kdd(isn, isn);
      
      % Gather all the matrices into one final matrix K which is the
      % training covariance matrix
      K = K+[Kff Kfd'; Kfd Kdd];
      
      [a b] = size(K);
    end    
  end  
  %add jitterSigma2 to the diagonal
  if ~isempty(gp.jitterSigma2)
    a1=a + 1;
    K(1:a1:end)=K(1:a1:end) + gp.jitterSigma2;
  end
  if nargout > 1
    C = K;
    if isfield(gp,'lik_mono') && isequal(gp.lik.type, 'Gaussian');
      % Add Gaussian noise to the obs part of covariance
      lik = gp.lik;
      Noi=lik.fh.trcov(lik, x1);
      Noi=[Noi zeros(size(Noi,1),size(K,2)-size(Noi,2)); ...
        zeros(size(K,1)-size(Noi,1),size(Noi,2)) ...
        zeros(size(K,1)-size(Noi,1),size(K,2)-size(Noi,2))];
      C=K+Noi;
%       x2=repmat(x1,m,1);
%       Cff = Kff + Noi;
%       C = [Cff Kfd'; Kfd Kdd];
    end
    if ~isempty(gp.jitterSigma2)
      C(1:a1:end)=C(1:a1:end) + gp.jitterSigma2;
    end
  end
end
