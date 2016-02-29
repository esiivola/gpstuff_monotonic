function gp = gp_monotonicbc(gp, varargin)
%GP_MONOTONICBC Converts GP structure to monotonic border condition GP and
%             optimizes the hyperparameters
%
% ---------[OUT OF DATE]---------->
%  Description
%    GP = GP_MONOTONICBC(GP, X, Y, OPTIONS) takes a GP structure GP
%    together with a matrix X of input vectors and a matrix Y of
%    target vectors and converts the GP structure to monotonic GP,
%    where the latent function is assumed to be monotonic w.r.t
%    input-dimensions.  In addition, this function can optimize the
%    hyperparameters of the GP (optimize='on'). Monotonicity is forced
%    by GP prior by adding virtual observations XV and YV where the
%    function derivative is assumed to be either positive (YV=1,
%    default) or negative (YV=-1). This function adds virtual
%    observations from the true observations two at a time until the
%    monotonicity is satisfied in every observation point. If GP
%    doesn't have field XV, the virtual observations are initialized
%    sampling from X or using K-means with K=floor(N/4), where N is
%    the number of true observations. Return monotonic GP structure GP
%    with optimized hyperparameters.
%
%    OPTIONS is optional parameter-value pair
%      z        - Optional observed quantity in triplet (x_i,y_i,z_i)
%                 Some likelihoods may use this. For example, in case of
%                 Poisson likelihood we have z_i=E_i, that is, expected
%                 value for ith case.
%      nv       - number of virtual observations to be used
%                 Default value is n/4
%      nvbd     - [Dimensions for which the latent functions is assumed to
%                 be monotonic. Use negative elements for monotonically
%                 decreasing and positive elements for monotonically
%                 increasing dimensions. Default 1:size(X,2), i.e.
%                 monotonically for all covariate dimensions.] UPDATE THIS
%      nu       - The strictness of the monotonicity information, with a 
%                 smaller values corresponding to the more strict information. 
%                 Default is 1e-6.
%      optimize - Option whether to optimize GP parameters. Default = 'off'. 
%      opt      - Options structure for optimizer.
%      optimf   - Function handle for an optimization function, which is
%                 assumed to have similar input and output arguments
%                 as usual fmin*-functions. Default is @fminscg.
%
%  See also
%    GP_SET
%
%  Reference
%    Riihimäki and Vehtari (2010). Gaussian processes with
%    monotonicity information.  Journal of Machine Learning Research:
%    Workshop and Conference Proceedings, 9:645-652.
%
% Copyright (c) 2014 Ville Tolvanen
% Copyright (c) 2015 Aki Vehtari
% Copyright (C) 2016 Eero Siivola
% <----------------\[OUT OF DATE]----------------------
%
% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

% parse inputs
ip=inputParser;
ip.FunctionName = 'GP_MONOTONIC';
ip.addRequired('gp',@isstruct);
ip.addOptional('x', [], @(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))))
ip.addOptional('y', [], @(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('z', [], @(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('nu', 1e-6, @(x) isreal(x) && isscalar(x) && (x>0))
ip.addParamValue('nv', [], @(x) isreal(x) && isscalar(x) && (x>=0))
ip.addParamValue('nvbd', [], @(x) isreal(x) && (size(x,1)<3));
ip.addParamValue('init', 'sample', @(x) ismember(x, {'sample', 'kmeans'}));
ip.addParamValue('optimf', @fminscg, @(x) isa(x,'function_handle'))
ip.addParamValue('opt', [], @isstruct)
ip.addParamValue('optimize', 'off', @(x) ismember(x, {'on', 'off'}));
ip.parse(gp, varargin{:});
x=ip.Results.x;
y=ip.Results.y; 
z=ip.Results.z;
nu=ip.Results.nu;
nv=ip.Results.nv;
init=ip.Results.init;
opt=ip.Results.opt;
optimf=ip.Results.optimf;
optimize=ip.Results.optimize;
nvbd=ip.Results.nvbd;
% Check appropriate fields in GP structure and modify if necessary to make
% proper monotonic GP structure
if ~isfield(gp, 'lik_mono') || ~ismember(gp.lik_mono.type, {'Probit', 'Logit'}) 
  gp.lik_mono=lik_probit();
end
if ~isfield(gp, 'lik_mono') || ~ismember(gp.lik_mono.type, {'Probit', 'Logit'}) 
  gp.lik_mono=lik_probit();
end
gp.derivobs=1;
% the strictness of the monotonicity information
gp.lik_mono.nu=nu;
% Set the virtual observations, here we use 25% of the observations as
% virtual points at initialization
if isempty(nv)
  frac=0.25;
  nv=floor(frac.*size(x,1));
end
if ~isempty(nvbd)
  gp.nvbd=nvbd;
else
  if isfield(gp, 'nvbd') && ~ismember('nvbd',ip.UsingDefaults(:)) 
  else
    if ~isfield(gp, 'nvbd')
        tmp = 1:size(x,2);
        gp.nvbd=[tmp;tmp];
    end
  end
end
if isempty(opt) || ~isfield(opt, 'TolX')
  % No options structure given or not a proper options structure
  opt=optimset('TolX',1e-4,'TolFun',1e-4,'Display','iter');
end
if ~isfield(gp,'latent_method') || ~strcmpi(gp.latent_method,'EP')
    fprintf('Switching the latent method to EP.\n');
    gp=gp_set(gp,'latent_method','EP');
end
gp.latent_opt.init_prev='off';
gp.latent_opt.maxiter=100;
gpep_e('clearcache',gp);
if isequal(optimize, 'on')
  % Optimize the parameters
  gp=gp_optim(gp,x,y,'opt',opt, 'z', z, 'optimf', optimf);
end

%Add parameter to the structure telling that there are border conditions 
gp.bc = 1;

end

