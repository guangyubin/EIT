
randn('seed', 0);
rand('seed',0);

m = 800;       % number of examples
n = 400;       % number of features
p = 100/n;      % sparsity density

x0 = sprandn(n,1,p);
A = randn(m,n);
A = A*spdiags(1./sqrt(sum(A.^2))',0,n,n); % normalize columns
b = A*x0 + sqrt(0.001)*randn(m,1);
% F = speye(n);
F = diag(abs(randn(1,n)));

lambda_max = norm( A'*b, 'inf' );
lambda = 0.1*lambda_max;


%%
  [x1, history2] = gllasso_ADMM(A, b,F,lambda, 1.0, 1.0);
 [x2, history2] = lasso_ADMM(A, b,lambda, 1.0, 1.0);
 tic
 [x3 k] = LpLqReg(A,F,b,2,1,lambda,1);
  toc  
   tic
  [x4 k] = LpLqReg(A,F,b,2,1,lambda,2);
  toc  
  figure;subplot(211);plot(x1);  hold on; plot(x3); legend('gllasso-ADMM','abs(L*x_est);')  
subplot(212);plot(x1); hold on;plot(x4); legend('gllasso-ADMM','abs(x_est)')

