function [h,w,alpha,strong_error,weak_error,F] = compute_h(h,w,alpha,strong_error,weak_error,F,features,N_pos,N_neg) 
    m = size(features,1);
    n = size(features,2);
    T = size(h,1);
    % Compute weighted error & choose weak learner
    [h_t weak_error]= find_h_t(w,weak_error,T,features,N_pos,N_neg); % h_t = [index, threshold, error, polarity]
    % Assign voting weight
    alpha_t = 1/2*log((1 - h_t(3))/h_t(3));
    % Return alpha_t & h_t
    alpha = [alpha;alpha_t];
    h = [h;h_t];
    % Compute strong classifier error
    T = size(h,1); % Update iteration T
    result = zeros(1,n);
    for i = 1:T
        result = result + alpha(i)*(features(h(i,1),:) > h(i,2))*h(i,4);
    end
    strong_error_t = (sum(result(1:N_pos) <= 0) + sum(result(N_pos:end) > 0))/(N_pos + N_neg);
    strong_error = [strong_error;strong_error_t];
    % Compute hisgrams over F(x)
    if (T == 10 || T == 50 || T == 100)
        F_t = zeros(1,n);
        for i = 1:T
            F_t = F_t + alpha(i)*(features(h(i,1),:) > h(i,2))*h(i,4);
        end
        F = [F;F_t];
    end
    % Update weights
    w(1:N_pos) = w(1:N_pos).*exp(-alpha_t*(features(h_t(1),1:N_pos)' >= h_t(2))*h_t(4));
    w(N_pos+1:end) = w(N_pos+1:end).*exp(-alpha_t*(features(h_t(1),N_pos+1:end)' <= h_t(2))*h_t(4));
    % Renormalize weights
    w = w/sum(w);
end