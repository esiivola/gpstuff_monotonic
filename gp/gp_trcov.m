function [K, C] = gp_trcov(gp, x1, predcf)
%GP_TRCOV  Evaluate training covariance matrix (gp_cov + noise covariance). 
%
%  Description
%    K = GP_TRCOV(GP, TX, PREDCF) takes in Gaussian process GP and
%    matrix TX that contains training input vectors to GP. Returns
%    (noiseless) covariance matrix K for latent values, which is
%    formed as a sum of the covariance matrices from covariance
%    functions in gp.cf array. Every element ij of K contains
%    covariance between inputs i and j in TX. PREDCF is an array
%    specifying the indexes of covariance functions, which are used
%    for forming the matrix. If not given, the matrix is formed
%    with all functions.
%
%    [K, C] = GP_TRCOV(GP, TX, PREDCF) returns also the (noisy)
%    covariance matrix C for observations y, which is sum of K and
%    diagonal term, for example, from Gaussian noise.
%
%  See also
%    GP_SET, GPCF_*
%
% Copyright (c) 2006-2010, 2016 Jarno Vanhatalo
% Copyright (c) 2010 Tuomas Nikoskinen

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

% no covariance functions?
if length(gp.cf)==0 || (nargin>2 && ~isempty(predcf) && predcf(1)==0) ...
        || isfield(gp, 'lik_mono')
    K=[];
    C=[];
    if nargout>1 && isfield(gp.lik.fh,'trcov')
        C=sparse(0);
        % Add Gaussian noise to the covariance
        C = C + gp.lik.fh.trcov(gp.lik, x1);
        if ~isempty(gp.jitterSigma2)
            C=C+gp.jitterSigma2;
        end
    end
    return
end

switch gp.type
    case {'FULL' 'FIC' 'PIC' 'PIC_BLOCK' 'CS+FIC' 'VAR' 'DTC' 'SOR'}
        [n,m]=size(x1);
        ncf = length(gp.cf);
        
        % Evaluate the covariance without noise
        K = sparse(0);
        K2 = sparse(0);
        if isfield(gp,'derivobs') && gp.derivobs  % derivative observations in use
            if any(strcmp(gp.type,{'FIC' 'PIC' 'PIC_BLOCK' 'CS+FIC' 'VAR' 'DTC' 'SOR'}))
                error('derivative observations have not been implemented for sparse GPs')
            end
        end
        % check whether predcf is used
        if nargin < 3 || isempty(predcf)
            predcf = 1:ncf;
        end
        % loop through covariance functions
        for i=1:length(predcf)
            gpcf = gp.cf{predcf(i)};
            if isfield(gp.lik, 'int_magnitude') && gp.lik.int_magnitude
                if ~isfield(gp,'comp_cf') || (isfield(gp,'comp_cf') && sum(gp.comp_cf{1}==predcf(i)))
                    gpcf.magnSigma2=1;
                end
            end
            % derivative observations in use
            if isfield(gp,'derivobs') && gp.derivobs
                if m==1
                    Kff = gpcf.fh.trcov(gpcf, x1);
                    Kdf = gpcf.fh.ginput4(gpcf, x1);
                    Kdf2 = gpcf.fh.ginput4(gpcf, gp.deriv_x_vals, x1);
                    Kdf2 = Kdf2{1}(gp.deriv_i,:);
                    D = gpcf.fh.ginput2(gpcf, x1, x1);
                    D2 = gpcf.fh.ginput2(gpcf, gp.deriv_x_vals, gp.deriv_x_vals);
                    D2{1} = D2{1}(gp.deriv_i, gp.deriv_i(:)); 
                    Kdf=Kdf{1};
                    Kfd = Kdf';
                    Kdd=D{1};
                    Kdd2 = D2{1};
                    
                    % Add all the matrices into a one K matrix
                    K = K + [Kff Kfd; Kdf Kdd];
                    K2 = K2 + [Kff Kdf2'; Kdf2 Kdd2];
                    K=K2;
                else                                                %UNTESTED 
                    % the block of covariance matrix
                    Kff = gpcf.fh.trcov(gpcf, x1);
                    % the blocks on the left side, below Kff 
                    Kdf= gpcf.fh.ginput4(gpcf, x1);
                    Kdf=cat(1,Kdf{1:m});
                    Kdf12= gpcf.fh.ginput4(gpcf, x1, gp.deriv_x_vals);
                    for i= 1:m % reduce the size of the matrices to correspond correct size of the derivative observations
                        Kdf12{i} = Kdf12{i}(:, gp.deriv_i(:,i));
                    end
                    Kdf12 = cat(1,Kdf12{:});
                    Kfd=Kdf';
                    % the diagonal blocks of double derivatives
                    D = gpcf.fh.ginput2(gpcf, x1, x1);
                    D2 = gpcf.fh.ginput2(gpcf, gp.deriv_x_vals, gp.deriv_x_vals);
                    % the off diagonal blocks of double derivatives on the
                    % upper right corner. See e.g. gpcf_squared -> ginput3
                    Kdf2 = gpcf.fh.ginput3(gpcf, x1 ,x1);
                    Kdf21 = gpcf.fh.ginput3(gpcf, gp.deriv_x_vals , gp.deriv_x_vals);                     
                    % Now build up Kdd m*n x m*n matrix, which contains all the
                    % both partial derivative" -matrices
                    
                    % Add the diagonal matrices
                    Kdd=blkdiag(D{1:m});
                    
                    

                    ns = [1:m];
                    for i= 1:m % reduce the size of the matrices to correspond correct size of the derivative observations
                        D2{i} = D2{i}(gp.deriv_i(:,i), gp.deriv_i(:,i));
                        ns(i) = sum(gp.deriv_i(:,i));
                    end
                    Kdd2 = blkdiag(D2{1:m});
                    
                    %OLD
                    % Add the non-diagonal matrices to Kdd
                    ii3=0;
                    for j=0:m-2
                        for i=1+j:m-1
                            ii3=ii3+1;
                            Kdd(i*n+1:(i+1)*n,j*n+1:j*n+n) = Kdf2{ii3}';
                            Kdd(j*n+1:j*n+n,i*n+1:(i+1)*n) = Kdf2{ii3};
                        end
                    end
                    %NEW
                    ii3=0;
                    csj=0;
                    for j=0:m2-2
                        csi = csj;
                        csj = csj + ns(j+1);
                        for i=1+j:m2-1
                            ii3=ii3+1;
                            Kdd2(csi+1:csi+ns(i),csj+1:csj+ns(j+2)) = Kdf21{ii3}';
                            Kdd2(csj+1:csj+ns(j+2),csi+1:csi+ns(i)) = Kdf21{ii3};
                            csi = csi + ns(i);
                        end
                    end
                                        
                    % Gather all the matrices into one final matrix K which is the
                    % training covariance matrix
                    K = K +  [Kff Kfd; Kdf Kdd];
                    K2 = K2 + [Kff2 Kdf12'; Kdf12 Kdd2];
                    K=K2;
                end
            else
                % Regular GP without derivative observations
                K = K + gpcf.fh.trcov(gpcf, x1);
            end
        end
        n = size(K,1);
        n1 = n+1;
        if ~isempty(gp.jitterSigma2)
            if issparse(K)
                K = K + sparse(1:n,1:n,gp.jitterSigma2,n,n);
            else
                K(1:n1:end)=K(1:n1:end) + gp.jitterSigma2;
            end
        end
        if nargout>1
            C=K;
            C1=K;
            if isfield(gp.lik.fh,'trcov')
                % Add Gaussian noise to the covariance
                if isfield(gp,'derivobs') && gp.derivobs  % derivative observations in use
                    % same noise for obs and grad obs
                    %C = C + gp.lik.fh.trcov(gp.lik, repmat(x1,m+1,1));
                    tmp = repmat(gp.deriv_x_vals,m,1);
                    C1 = C1 + gp.lik.fh.trcov(gp.lik, [x1; tmp(gp.deriv_i(:),:)]);
                    C = C1;
                else
                    C = C + gp.lik.fh.trcov(gp.lik, x1);
                end
                
            end
        end
        
end

