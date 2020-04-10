

function [x_est, k] = LpLqReg(A,L,b,p,q,lambda , pp)

if nargin < 7
    pp=1;
end
epsf = 1e-6;
tol = 1e-3;
diff = 1e3;
MaxIter = 1000;
k = 1;
[m n] = size(A);

x_est = (A'*A + lambda*L'*L)\(A'*b);
% x_est = zeros(n, 1);
% x_est = A'*b;
% ATA = A'*A;
singularTV = 0;
while (diff > tol && singularTV == 0 )
    %     z = A*x_est - b + lambda*L*x_est;
    %     beta(k) = norm(z);
    
    v = abs(A*x_est-b);
    if pp ==1
	 z = abs(L*x_est);
    else
    z = abs(x_est);
    end
    
    mask_v = (v >= epsf);
    mask_z = (z >= epsf);
    v = (mask_v.*v) + (1-mask_v)*epsf;
    z = (mask_z.*z) + (1-mask_z)*epsf;

    wf = power( v, p-2);
    wr = power( z, q-2);

    
    WF = sparse(diag(wf));
    WR = sparse(diag(wr));

%     WF = (diag(wf));
%     WR = (diag(wr));
    
%     x_est_new = inv(A'*WF*A+ lambda*L'*WR*L)*A'*WF*b;
    x_est_new = (A'*WF*A + lambda*L'*WR*L)\(A'*WF*b);
    difftmp = norm(x_est_new - x_est)/norm(x_est);
    
    dd_diff_temp = abs(difftmp - diff);
    loss(k) = difftmp;
    if(difftmp < diff)
        diff = difftmp;
    else
        break;
    end
    k = k+1;
    x_est = x_est_new;
    if(k > MaxIter)
        break;
    end
end
a = 0;
disp( ['LpLqReg iter = ', num2str(k) ] );
end
