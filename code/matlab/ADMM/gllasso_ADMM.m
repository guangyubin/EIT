function [x, history] = gllasso_ADMM(A, b,F, lambda, rho, alpha)
% lasso  Solve Generalized lasso problem via ADMM
%
% [z, history] = lasso(A, b, F,lambda, rho, alpha);
% A: m*n  b m*1  x n*1   F m2*n 
% Solves the following problem via ADMM:
%
%   minimize 1/2*|| Ax - b ||_2^2 + \lambda || Fx ||_1
%
% The solution is returned in the vector x.
%
% history is a structure that contains the objective value, the primal and
% dual residual norms, and the tolerances for the primal and dual residual
% norms at each iteration.
%
% rho is the augmented Lagrangian parameter.
%
% alpha is the over-relaxation parameter (typical values for alpha are
% between 1.0 and 1.8).
%
%
% More information can be found in the paper linked at:
% http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
%

t_start = tic;

%% Global constants and defaults

QUIET    = 0;
MAX_ITER = 1000;
ABSTOL   = 1e-4;
RELTOL   = 1e-2;

%% Data preprocessing

[m, n] = size(A);

% save a matrix-vector multiply
Atb = A'*b;

FtF = F'*F;


%% ADMM solver
[m2 ,n2] = size(F);
x = zeros(n,1);
z = zeros(m2,1);
u = zeros(m2,1);

% cache the factorization
[L, U] = factor(A,FtF, rho);
%  ZZ = A'*A + rho*FtF;
if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
        'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

for k = 1:MAX_ITER
    
    % x-update
    q = Atb + rho*F'*(z - u);    % temporary value
%     if( m >= n )    % if skinny
%         x = U \ (L \ q);
%     else            % if fat
%         x = q/rho - (A'*(U \ ( L \ (A*q) )))/rho^2;
%     end
        x = U \ (L \ q);
%     x =  ZZ\q;
    % z-update with relaxation
    zold = z;
    x_hat = alpha*F*x + (1 - alpha)*zold;
    z = shrinkage(x_hat + u, lambda/rho);
    
    % u-update
    u = u + (x_hat - z);
    
    % diagnostics, reporting, termination checks
    history.objval(k)  = objective(A, b, lambda, x, z);
    
    history.r_norm(k)  = norm(F*x - z);
    history.s_norm(k)  = norm(-rho*F'*(z - zold));
    
    history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(F*x), norm(-z));
    history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*F'*u);
    
    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end
    
    if (history.r_norm(k) < history.eps_pri(k) && ...
            history.s_norm(k) < history.eps_dual(k))
        break;
    end
    
end

if ~QUIET
    toc(t_start);
end

end

function p = objective(A, b, lambda, x, z)
p = ( 1/2*sum((A*x - b).^2) + lambda*norm(z,1) );
end

function z = shrinkage(x, kappa)
z = max( 0, x - kappa ) - max( 0, -x - kappa );
end

function [L U] = factor(A,FtF, rho)
[m, n] = size(FtF);
% [m2 ,n ] = size(FtF);
% if ( m >= n )    % if skinny
%     L = chol( A'*A + rho*FtF, 'lower' );
% else            % if fat
%     L = chol( speye(m) + 1/rho*(A*A'), 'lower' );
% end
L = chol( A'*A + rho*FtF, 'lower' );
% force matlab to recognize the upper / lower triangular structure
L = sparse(L);
U = sparse(L');
end
