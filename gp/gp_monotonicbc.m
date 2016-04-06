function gp = gp_monotonicbc(gp, varargin)
%GP_MONOTONIC Converts GP structure to monotonic GP and optimizes the
%             hyperparameters
%
%  Description
%    GP = GP_MONOTONIC(GP, X, Y, OPTIONS) takes a GP structure GP
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
%      nv       - Number of virtual observations to be used at initialization.
%                 Default value is floor(n/4) where n is the number of observations.
%      nvbd      - Dimensions for which the latent functions is assumed to
%                 be monotonic. Use negative elements for monotonically
%                 decreasing and positive elements for monotonically
%                 increasing dimensions. Default [1:size(X,2)'; 1:size(X,2)'], i.e.
%                 monotonically for all covariate dimensions in both ends.
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
% Copyright (c) 2016 Eero Siivola

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
ip.addParamValue('nv', [], @(x) isreal(x) && isscalar(x))
ip.addParamValue('optimf', @fminscg, @(x) isa(x,'function_handle'))
ip.addParamValue('opt', [], @isstruct)
ip.addParamValue('optimize', 'off', @(x) ismember(x, {'on', 'off'}));
ip.addParamValue('nvbd', [], @(x) isreal(x));
ip.parse(gp, varargin{:});
x=ip.Results.x;
y=ip.Results.y;
z=ip.Results.z;
if (ip.Results.nv> size(x,1)) nv = size(x,1); else nv = ip.Results.nv;end;
opt=ip.Results.opt;
optimf=ip.Results.optimf;
optimize=ip.Results.optimize;
nvbd=ip.Results.nvbd;
% Check appropriate fields in GP structure and modify if necessary to make
% proper monotonic GP structure
if ~isfield(gp, 'lik_mono') || ~ismember(gp.lik_mono.type, {'Probit', 'Logit'}) 
 gp.lik_mono=lik_probit();
end
% Set the virtual observations, here we use 25% of the observations as
% virtual points at initialization
if isempty(nv)
  frac=0.1;
  nv=floor(frac.*size(x,1));
else
  gp.nv=nv;
end
if ~isempty(nvbd)
  gp.nvbd=nvbd;
else
  if isfield(gp, 'nvbd') && ~ismember('nvbd',ip.UsingDefaults(:)) 
  else
    if ~isfield(gp, 'nvbd')
      gp.nvbd=1:size(x,2);
      gp.nvbd = [gp.nvbd(:)'; gp.nvbd(:)'];
    end
  end
end

% Find round(nv/2) smallest and round(size(y)-nv/2) largest in each dimension in nvbd
nl = round(gp.nv/2); nh = gp.nv - round(gp.nv/2);
xv=[];
yv=[];
m=size(x, 2);
for i=1:length(gp.nvbd(1,:))
  yl = zeros(nl, m);
  yh = zeros(nh, m);
  j = abs(gp.nvbd(1,i)); % Current dimension
  lv = gp.nvbd(1,i)/j; % Direction of partial derivative at low values
  hv = gp.nvbd(2,i)/abs(gp.nvbd(2,i)); % Direction of partial derivative at high values
  [~,ind(:,j)]=sort(x(:,j),'ascend'); % Sort training sample values in this direction
  lit = ind(1:nl,:); hit = ind(end-nh+1:end, :); % find indices of nl smallest and nh largest
  xv = [xv; x(lit,:); x(hit,:)];
  yl(:,j)= lv*ones(nl,1);
  hl(:,j)= hv*ones(nh,1);
  yv=[yv;yl;hl];
end
gp = gp_der(gp, xv, yv, logical(yv));

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
end

