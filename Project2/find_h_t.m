function [h_t weak_error]= find_h_t(w,weak_error,T,features,N_pos,N_neg)
    % Given weighted features, return the best filter
    m = size(features,1);
    n = size(features,2);
    params = [min(features,[],2) max(features,[],2) 0.5*ones(m,2)];
    params = [params (params(:,2) - params(:,1))/25];
    % paras = [argmin, argmax, minprob, maxprob, step]
    % default polarity = +1
    threshold = params(:,1);
    for cnt = 1:25 % step = (max - min)/1000
        threshold = threshold + params(:,5);
        prob = sum([w(1:N_pos)'.*(features(:,1:N_pos) < threshold) w(N_pos+1:end)'.*(features(:,N_pos+1:end) >= threshold)],2);
        for i = 1:m 
            if prob(i) < params(i,3)
                params(i,3) = prob(i);
                params(i,1) = threshold(i);
            end
            if prob(i) > params(i,4)
                params(i,4) = prob(i);
                params(i,2) = threshold(i);
            end
        end
    end
    % index, threshold, error_prob, polarity
    all = zeros(m,3);
    for i = 1:m
        if params(i,3) < 1 - params(i,4)
            all(i,1) = params(i,1);
            all(i,2) = params(i,3);
            all(i,3) = 1;
        else 
            all(i,1) = params(i,2);
            all(i,2) = 1 - params(i,4);
            all(i,3) = -1;
        end
    end
    [~,index] = min(all(:,2));
    % index, threshold, error, polarity
    h_t = [index all(index,:)];
    
    % Compute weak classifier error
    if (T == 0 || T == 10 || T == 50 || T == 100)
        tmp_weak_error = sort(all(:,2));
        weak_error = [weak_error;tmp_weak_error(1:1000)'];
    end
end