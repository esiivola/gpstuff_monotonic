%DEMO_BAYESIANOPTIMIZATION  A demonstration program for Bayesian
%                           optimization
%
%  Part 1:
%  One dimensional example
%
%  Part 2:
%  One dimensional example 
%
%  Part 3:
%  Two dimensional example 
% 
%  References:
%    Jones, D., Schonlau, M., & Welch, W. (1998). Efficient global
%    optimization of expensive black-box functions. Journal of Global
%    Optimization, 13(4), 455-492. doi:10.1023/a:1008306431147  
%
%    Michael A. Gelbart, Jasper Snoek, and Ryan P. Adams
%    (2014). Bayesian Optimization with Unknown Constraints.
%    http://arxiv.org/pdf/1403.5607v1.pdf
%
%    Snoek, J., Larochelle, H, Adams, R. P. (2012). Practical Bayesian
%    Optimization of Machine Learning Algorithms. NIPS 25 
%
%  Copyright (c) 2015 Jarno Vanhatalo
%                2016 Eero Siivola

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

%%  Part 1:
%  One dimensional example 
clear all;
close all;

% Construct a function to be optimized
xl = linspace(0,10,100)';
fx = @(x) 0.6*x -0.1*x.^2 + sin(2*x)+ 3*exp(-1.2*x) + exp(-(10-x));

bcx = [0;10]; % Points for virtual derivative observations
bcy = [-1;1]; % Virtual derivative observations for corresponding points

% construct GP
cfc = gpcf_constant('constSigma2',1,'constSigma2_prior', prior_fixed);
cfse = gpcf_sexp('lengthScale',2,'magnSigma2',1,'magnSigma2_prior',prior_sqrtt('s2',10));
lik = lik_gaussian('sigma2', 0.01, 'sigma2_prior', prior_fixed);
%lik = lik_gaussian('sigma2', 1);
gp = gp_set('cf', {cfc, cfse}, 'lik', lik);

% gpcf=gpcf_sexp('lengthScale',1,'magnSigma2',1,'magnSigma2_prior',prior_sqrtt('s2',10));
% lik=lik_gaussian();

%gp=gp_set('cf', {gpcf, gpcf_constant}, 'lik', lik, 'jitterSigma2', 1e-9);
gp0=gp;

gp = gp_der(gp, bcx, bcy); % Add derivative observations to the model
gp.lik_mono=lik_probit();
gp=gp_set(gp,'latent_method','EP');
% % the strictness of the monotonicity information
gp.lik_mono.nu=1e-9; %gp.lik_mono.nu=1;
% ----- conduct Bayesian optimization -----
% draw initial point
gp.latent_opt.init_prev='off';
gp.latent_opt.maxiter=100;
gpep_e('clearcache',gp);


% Set the options for optimizer of the acquisition function
optimf = @fmincon;
optdefault=struct('GradObj','on','LargeScale','off','Algorithm','SQP','TolFun',1e-6,'TolX',1e-3);
%opt=optimset(optdefault);
opt=optimset('TolX',1e-4,'TolFun',1e-4); %'Display','iter');
lb=0;     % lower bound of the input space
ub=10;    % upper bound of the input space

% draw initial point
rng(3)
x = 10*rand;
y = fx(x);

gp0.lik.sigma2=1;subplot(211),rng(1);fs=gp_rnd(gp0,x,y,xl,'nsamp',10);plot(xl,fs)
gp.lik.sigma2=1;subplot(212),gpep_e('clearcache',gp);rng(1);fs=gp_rnd(gp,x,y,xl,'nsamp',10);plot(xl,fs)
gp.lik.sigma2=0.001;

% draw initial point
rng(3)
x = 10*rand;
y = fx(x);
gp.lik_mono.nu=1e-6; %gp.lik_mono.nu=1;

i1 = 1;
maxiter = 15;
improv = inf;   % improvement between two successive query points
while i1 < maxiter && improv>1e-6
%while i1 < maxiter

    % Train the GP model for objective function and calculate variables
    % that are needed when calculating the Expected improvement
    % (Acquisition function) 
    if i1>1
        gp = gp_optim(gp,x,y,'opt',opt, 'optimf', @fminscg);
        [gpia,pth,th]=gp_ia(gp,x,y);%,'int_method','grid');
        gp = gp_unpak(gp,sum(bsxfun(@times,pth,th)));
    end
    [K, C] = gp_trcov(gp,x);
    invC = inv(C);
    a = C\[y; bcy];
    fmin = min( fx(x) );
    
    % Calculate EI and posterior of the function for visualization purposes
    [Ef,Varf] = gp_pred(gp, x, y, xl); %gp_pred(gp, x, [y; bcy], xl); 
    Ef=Ef(1:size(xl,1));
    Varf=Varf(1:size(xl,1));
    EI = expectedimprovement_e(xl, gp, x, Ef, Varf, fmin);
    
     x_new = xl(find(EI==min(EI)),:);
    
    % optimize acquisition function
    %    Note! Opposite to the standard notation we minimize negative Expected
    %    Improvement since Matlab optimizers seek for functions minimum
    % Here we use multiple starting points for the optimization so that we
    % don't crash into suboptimal mode
    % fh_eg = @(x_new) expectedimprovement_eg(x_new, gp, x, a, invC, fmin); % The function handle to the Expected Improvement function
    % indbest = find(y == fmin);
    % xstart = [linspace(0.5,9.5,5) x(indbest)+0.1*randn(1,2)];
    % for s1=1:length(xstart)
    %     x_new(s1) = optimf(fh_eg, xstart(s1), [], [], [], [], lb, ub, [], opt);
    % end
    % EIs = expectedimprovement_eg(x_new(:), gp, x, a, invC, fmin);    
    % x_new = x_new( find(EIs==min(EIs),1) ); % pick up the point where Expected Improvement is maximized
        
    % put new sample point to the list of evaluation points
    x(end+1) = x_new;
    y(end+1) = fx(x(end));  % calculate the function value at query point
    x=x(:);y=y(:);

    % visualize
    %    figure(1);
    clf
    subplot(2,1,1),hold on, title('function to be optimized and GP fit')
    %plot(xl,fx(xl))
    box on
    plot(xl,fx(xl),'r')
    % The function evaluations so far
    plot(x(1:end-1),y(1:end-1), 'ko')
    % The new sample location
    plot(x(end),y(end), 'ro')
    % the posterior of the function
    plot(xl,Ef, 'k')
    plot(xl,Ef + 2*sqrt(Varf), 'k--')
    plot(xl,Ef - 2*sqrt(Varf), 'k--')
    legend('objective function', 'function evaluations', 'next query point', 'GP mean', 'GP 95% interval','location','southwest')
    % The expected information    
    subplot(2,1,2)
    plot(xl,EI, 'r'), hold on
    plot(x(end),0, 'r*')
    plot(x(end)*[1 1],ylim, 'r--')
    title('acquisition function')
    % figure(2)
    % clf;
    % [fr] = gp_rnd(gp, x(1:end-1,:), y(1:end-1,:), xl,'nsamp',10);
    % plot(xl,fr)
    %    figure(1)   
    improv = abs(y(end) - y(end-1));
    i1=i1+1;
    pause
end

%%  Part 2:
%  One dimensional example 
clear

% Construct a function to be optimized
xl = linspace(0,10,100)';
fx = @(x) 0.6*x -0.1*x.^2 + sin(2*x);

bcx = [0;10]; % Points for virtual derivative observations
bcy = [-1;1]; % Virtual derivative observations for corresponding points

% construct GP
cfc = gpcf_constant('constSigma2',10,'constSigma2_prior', prior_fixed);
cfse = gpcf_sexp('lengthScale',1,'magnSigma2',1,'magnSigma2_prior',prior_sqrtt('s2',10));
lik = lik_gaussian('sigma2', 0.001, 'sigma2_prior', prior_fixed);
%lik = lik_gaussian();
gp = gp_set('cf', {cfc, cfse}, 'lik', lik);

gpcf=gpcf_sexp('lengthScale',1,'magnSigma2',1,'magnSigma2_prior',prior_sqrtt('s2',10));
lik=lik_gaussian();

gp=gp_set('cf', {gpcf, gpcf_constant}, 'lik', lik, 'jitterSigma2', 1e-9);


gp = gp_der(gp, bcx, bcy); % Add derivative observations to the model
gp.lik_mono=lik_probit();
gp=gp_set(gp,'latent_method','EP');
% % the strictness of the monotonicity information
gp.lik_mono.nu=10;
% ----- conduct Bayesian optimization -----
% draw initial point
gp.latent_opt.init_prev='off';
gp.latent_opt.maxiter=100;
gpep_e('clearcache',gp);

% Set the options for optimizer of the acquisition function
optimf = @fmincon;
optdefault=struct('GradObj','on','LargeScale','off','Algorithm','SQP','TolFun',1e-6,'TolX',1e-3);
%opt=optimset(optdefault);
opt=optimset('TolX',1e-4,'TolFun',1e-4); %'Display','iter');
lb=0;     % lower bound of the input space
ub=10;    % upper bound of the input space

% draw initial point
rng(3)
x = 10*rand;
y = fx(x);

i1 = 1;
maxiter = 15;
improv = inf;   % improvement between two successive query points
while i1 < maxiter && improv>1e-6
%while i1 < maxiter

    % Train the GP model for objective function and calculate variables
    % that are needed when calculating the Expected improvement
    % (Acquisition function) 
    if i1>1
        gp = gp_optim(gp,x,y,'opt',opt, 'optimf', @fminscg);
        %gp=gp_optim(gp,x,y,'opt',opt, 'optimf', @fminlbfgs);
    end
    [K, C] = gp_trcov(gp,x);
    invC = inv(C);
    a = C\[y; bcy];
    fmin = min( fx(x) );
    
    % Calculate EI and posterior of the function for visualization purposes
    EI = expectedimprovement_eg(xl, gp, x, a, invC, fmin);
    
    [Ef,Varf] = gp_pred(gp, x, y, xl); %gp_pred(gp, x, [y; bcy], xl); 
    Ef=Ef(1:size(xl,1));
    Varf=Varf(1:size(xl,1));
    
    % optimize acquisition function
    %    Note! Opposite to the standard notation we minimize negative Expected
    %    Improvement since Matlab optimizers seek for functions minimum
    % Here we use multiple starting points for the optimization so that we
    % don't crash into suboptimal mode
    fh_eg = @(x_new) expectedimprovement_eg(x_new, gp, x, a, invC, fmin); % The function handle to the Expected Improvement function
    indbest = find(y == fmin);
    xstart = [linspace(0.5,9.5,5) x(indbest)+0.1*randn(1,2)];
    for s1=1:length(xstart)
        x_new(s1) = optimf(fh_eg, xstart(s1), [], [], [], [], lb, ub, [], opt);
    end
    EIs = expectedimprovement_eg(x_new(:), gp, x, a, invC, fmin);    
    x_new = x_new( find(EIs==min(EIs),1) ); % pick up the point where Expected Improvement is maximized
        
    % put new sample point to the list of evaluation points
    x(end+1) = x_new;
    y(end+1) = fx(x(end));  % calculate the function value at query point
    x=x(:);y=y(:);

    % visualize
    figure(3)  
    clf
    subplot(2,1,1),hold on, title('function to be optimized and GP fit')
    %plot(xl,fx(xl))
    box on
    plot(xl,fx(xl),'r')
    % The function evaluations so far
    plot(x(1:end-1),y(1:end-1), 'ko')
    % The new sample location
    plot(x(end),y(end), 'ro')
    % the posterior of the function
    plot(xl,Ef, 'k')
    plot(xl,Ef + 2*sqrt(Varf), 'k--')
    plot(xl,Ef - 2*sqrt(Varf), 'k--')
    legend('objective function', 'function evaluations', 'next query point', 'GP mean', 'GP 95% interval','location','southwest')
    % The expected information    
    subplot(2,1,2)
    plot(xl,EI, 'r'), hold on
    plot(x(end),0, 'r*')
    plot(x(end)*[1 1],ylim, 'r--')
    title('acquisition function')
    figure(4)
    clf;
    [fr] = gp_rnd(gp, x, y, xl,'nsamp',10);
    plot(xl,fr) 
    improv = abs(y(end) - y(end-1));
    i1=i1+1;
    pause
end

%%  Part 2:
%  Two dimensional example 
clear

% The objective function
fx = @(x) -log( (mvnpdf([x(:,1) x(:,2)],[-1.5 -2.5], [1 0.3; 0.3 1]) + 0.3*mvnpdf([x(:,1) x(:,2)],[2 3], [3 0.5; 0.5 4])).*...
    mvnpdf([x(:,1) x(:,2)],[0 0], [100 0; 0 100])) ./15 -1;
% 
% fx = @(x) -log( (mvnpdf([x(:,1) x(:,2)],[-2 -2], [0.8 0.3; 0.3 0.8]) + 0.3*mvnpdf([x(:,1) x(:,2)],[2 5], [3 0.5; 0.5 4])).*...
%     mvnpdf([x(:,1) x(:,2)],[0 0], [100 0; 0 100])) ./15 -1 + 0.3*mvnpdf([x(:,1) x(:,2)],[5 2], [3 0.5; 0.5 4]);

% Help variables for visualization
lb=-5;
ub=5;
n=3;
g = linspace(lb+(ub-lb)/(n+1), ub-(ub-lb)/(n+1), n);
lbx1 = [repmat(lb,1, n); g]';
lbx2 = [g; repmat(lb,1, n)]';
ubx1 = [repmat(ub,1, n); g]';
ubx2 = [g; repmat(ub,1, n)]';

xd = [lbx1; lbx2; ubx1; ubx2];
yd = [repmat(-1, n, 1) repmat(0, n, 1);
      repmat(0, n, 1) repmat(-1, n, 1);
      repmat(1, n, 1) repmat(0, n, 1);
      repmat(0, n, 1) repmat(1, n, 1)];

[X,Y] = meshgrid(linspace(lb,ub,100),linspace(lb,ub,100));
xl = [X(:) Y(:)];
Z = reshape(fx(xl),100,100);

% construct GP to model the function
cfc = gpcf_constant('constSigma2',10,'constSigma2_prior', prior_fixed);
cfl = gpcf_linear('coeffSigma2', .01, 'coeffSigma2_prior', prior_sqrtt()); 
cfl2 = gpcf_squared('coeffSigma2', .01, 'coeffSigma2_prior', prior_sqrtt(), 'interactions', 'on');
cfse = gpcf_sexp('lengthScale',2,'lengthScale_prior',prior_loggaussian('mu',log(2),'s2',1),'magnSigma2',.1,'magnSigma2_prior',prior_loggaussian('mu',0,'s2',1));
%cfse = gpcf_sexp('lengthScale',2,'lengthScale_prior',prior_t('s2',1^2),'magnSigma2',.1,'magnSigma2_prior',prior_sqrtt('s2',10));
lik = lik_gaussian('sigma2', 0.0001, 'sigma2_prior', prior_fixed);
%lik = lik_gaussian('sigma2', 0.001, 'sigma2_prior', prior_sqrtt('s2',10));
%gp = gp_set('cf', {cfc, cfl, cfl2, cfse}, 'lik', lik);

gp = gp_set('cf', {cfc, cfse}, 'lik', lik);

if 1
gp = gp_der(gp, xd, yd, logical(yd));
if ~isfield(gp, 'lik_mono') || ~ismember(gp.lik_mono.type, {'Probit', 'Logit'}) 
 gp.lik_mono=lik_probit();
end
%the strictness of the monotonicity information
gp.lik_mono.nu=1e-6;
%gp.lik_mono.nu=1e6;
gp=gp_set(gp,'latent_method','EP');


% ----- conduct Bayesian optimization -----
% draw initial point
gp.latent_opt.init_prev='off';
gp.latent_opt.maxiter=100;
gpep_e('clearcache',gp);
end

% Set the options for optimizer of the acquisition function
optimf = @fmincon;
optdefault=struct('GradObj','on','LargeScale','off','Algorithm','trust-region-reflective','TolFun',1e-9,'TolX',1e-6);
opt=optimset('TolX',1e-4,'TolFun',1e-4);
lb=[-5 -5];     % lower bound of the input space
ub=[5 5];   % upper bound of the input space

% draw initial points
x = [-4 -4;-4 4;4 -4;4 4;0 0];
x = [-3 -3;-3 3;3 -3;3 3];
x = [-2 -2;-2 2;2 -2;2 2];
%x = [0 0];
y = fx(x);

%figure, % figure for visualization
i1 = 1;
maxiter = 20;
improv = inf;   % improvement between two successive query points
while i1 < maxiter %&& improv>1e-9
%while i1 < maxiter

    % Train the GP model for objective function and calculate variables
    % that are needed when calculating the Expected improvement
    % (Acquisition function) 
    if i1>1
        gp = gp_optim(gp,x,y);
        %gp = gp_optim(gp,x,y,'opt',opt, 'optimf', @fminlbfgs);
        %gp = gp_optim(gp,x,[y; gp.deriv_y_vals(gp.deriv_i)]);
        [gpia,pth,th]=gp_ia(gp,x,y);
        gp = gp_unpak(gp,sum(bsxfun(@times,pth,th)));
    end
    % [K, C] = gp_trcov(gp,x);
    % invC = inv(C);
    % a = C\[y; gp.deriv_y_vals(gp.deriv_i)];
    fmin = min( y );
    
    % Calculate EI and the posterior of the function for visualization
    [Ef,Varf] = gp_pred(gp, x, y, xl);
    EI = expectedimprovement_e(xl, gp, x, Ef, Varf, fmin);
    
    % optimize acquisition function
    %  * Note! Opposite to the standard notation we minimize negative Expected
    %    Improvement since Matlab optimizers seek for functions minimum
    %  * Note! We alternate the acquisition function between Expected
    %    Improvement and expected variance. The latter helps the
    %    optimization so that it does not get stuck in local mode
    % Here we use multiple starting points for the optimization so that we
    % don't crash into suboptimal mode of acquisition function

    % if mod(i1,4)==0
    %  x_new = xl(find(Varf==max(Varf)),:);
    % else
    [~,mini]=min(EI+double(ismember(xl,x,'rows'))*1e9);%+rand(size(EI))*1e-2);
     x_new = xl(mini,:);
    % end
    
    % put new sample point to the list of evaluation points
    x(end+1,:) = x_new;
    y(end+1,:) = fx(x(end,:));     % calculate the function value at query point

    % visualize
    clf
    % Plot the objective function
    subplot(2,2,1),hold on, title('Objective, query points')
    box on
    
    pcolor(X,Y,Z),shading flat
    clim = caxis;
    l1=plot(x(1:end-1,1),x(1:end-1,2), 'rx', 'MarkerSize', 10);
    %plot(x(end,1),x(end,2), 'ro', 'MarkerSize', 10)
    l3=plot(x(end,1),x(end,2), 'ro', 'MarkerSize', 10, 'linewidth', 3);
    legend([l1,l3], {'function evaluation points','The next query point'})
    % Plot the posterior mean of the GP model for the objective function
    subplot(2,2,2),hold on, title(sprintf('GP prediction, mean, iter: %d',i1))
    box on
    pcolor(X,Y,reshape(Ef(1:size(xl,1)),100,100)),shading flat
    caxis(clim)
    l3=plot(x(end,1),x(end,2), 'ro', 'MarkerSize', 10, 'linewidth', 3);
    % Plot the posterior variance of GP model
    subplot(2,2,4),hold on, title('GP prediction, variance')
    box on
    pcolor(X,Y,reshape(Varf(1:size(xl,1)),100,100)),shading flat
    %l2=plot(xnews(:,1),xnews(:,2), 'ro', 'MarkerSize', 10);
    l3=plot(x(end,1),x(end,2), 'ro', 'MarkerSize', 10, 'linewidth', 3);
    % Plot the expected improvement 
    subplot(2,2,3), hold on, title(sprintf('Expected improvement %.2e', min(EI)))
    box on
    pcolor(X,Y,reshape(EI,100,100)),shading flat
    %plot(xnews(:,1),xnews(:,2), 'ro', 'MarkerSize', 10);
    plot(x(end,1),x(end,2), 'ro', 'MarkerSize', 10, 'linewidth', 3);
   
       
    improv = abs(y(end) - y(end-1));
    i1=i1+1;
    [p, n] = gp_pak(gp);
    n
    exp(p)
    pause
end









% mu1=[-1.5 -2.5]; Sigma1=[1 0.3; 0.3 1];
% mu2=[2 3];Sigma2=[3 0.5; 0.5 4];
% mu3=[0 0];Sigma3=[100 0; 0 100];
% dfx = @(x) -1./( (mvnpdf(x,mu1,Sigma1) + 0.3*mvnpdf(x,mu2,Sigma2)).*mvnpdf(x,mu3,Sigma3))/15.*...
%     ( ( -mvnpdf(x,mu1,Sigma1).*(x-mu1)/Sigma1 - 0.3*mvnpdf(x,mu2,Sigma2).*(x-mu2)/Sigma2 ).*mvnpdf(x,mu3,Sigma3) -...
%      (mvnpdf(x,mu1,Sigma1) + 0.3*mvnpdf(x,mu2,Sigma2)).*mvnpdf(x,mu3,Sigma3).*(x-mu3)/Sigma3  ) ;
% for i = 1:size(xd,1)
%   yd(i,:) = dfx(xd(i,:));
% end
% id = logical([repmat(-1, n, 1) repmat(0, n, 1);
%       repmat(0, n, 1) repmat(-1, n, 1);
%       repmat(1, n, 1) repmat(0, n, 1);
%       repmat(0, n, 1) repmat(1, n, 1)]);


    %if mod(i1,5)==0  % Do just exploration by finding the maimum variance location
%     %   fh_eg = @(x_new) expectedvariance_eg(x_new, gp, x, [], invC);
%     %else
%     fh_eg = @(x_new) expectedimprovement_eg(x_new, gp, x, a, invC, fmin);
%     %end
%     indbest = find(y == fmin);
%     nstarts = 40;
%     xstart = [repmat(lb,nstarts,1) + repmat(ub-lb,nstarts,1).*rand(nstarts,2) ]; 
%     for s1=1:length(xstart)
%         x_new(s1,:) = optimf(fh_eg, xstart(s1,:), [], [], [], [], lb, ub, [], opt);
%     end
%     xnews = x_new;
%     EIs = expectedimprovement_eg(x_new, gp, x, a, invC, fmin);
%     x_new = x_new( find(EIs==min(EIs),1), : ); % pick up the point where Expected Improvement is maximized