function [stats, all_pos_edges, all_neg_edges] = runCPM(x,y,kfolds,age,sex)

% This function will run CPM on a measure of interest
%
% Input variables:
%  x - a 3D node x node x subject connectivity matrix
%  y - a vector of some phenotype to be predicted
%  kfolds - the number of cross-validation folds you want to use
%  age - the numeric age of the subjects (to be regressed out)
%  sex - the sex of the subjects (to be regressed out). MUST BE NUMERIC!
%
% Outputs:
%  stats - a struct containing stats that summarize predictive model
%          performance (Pearson and Spearman correlation coefficients and
%          p-values, MSE, and q_s)
%  all_pos_edges - an edge x kfolds matrix indicating which edges were
%          positively correlated with the measure of interest
%  all_neg_edges - an edge x kfolds matrix indicating which edges were
%          negatively correlated with the measure of interest
%
% Author: Alexander J Simon

N = length(y);

for s = 1:N
    all_edges(:, s) = squareform(tril(x(:, :, s), -1));
end

indices = cvpartition(N,'k',kfolds);

for i_fold = 1:kfolds

    test.indx = indices.test(i_fold);
    train.indx = indices.training(i_fold);

    test.x = all_edges(:,test.indx);
    train.x = all_edges(:,train.indx);

    test.y = y(indices.test(i_fold),:);
    train.y = y(indices.training(i_fold),:);

    test.age = age(indices.test(i_fold),:);
    train.age = age(indices.training(i_fold),:);

    test.sex = sex(indices.test(i_fold),:);
    train.sex = sex(indices.training(i_fold),:);

    % univariate edge selection
    if ~isempty(age) || ~isempty(sex)
        [edge_corr_age, edge_p_age] = partialcorr(train.x', train.y, train.age);
        [edge_corr_sex, edge_p_sex] = partialcorr(train.x', train.y, train.sex);

        edges_pos_age = (edge_p_age < 0.05) & (edge_corr_age > 0);
        edges_neg_age = (edge_p_age < 0.05) & (edge_corr_age < 0);

        edges_pos_sex = (edge_p_sex < 0.05) & (edge_corr_sex > 0);
        edges_neg_sex = (edge_p_sex < 0.05) & (edge_corr_sex < 0);

        edges_pos = (edges_pos_age == 1) & (edges_pos_sex == 1);
        edges_neg = (edges_neg_age == 1) & (edges_neg_sex == 1);

    else
        [edge_corr, edge_p] = corr(train.x', train.y,'rows','complete');

        edges_pos = (edge_p < 0.05) & (edge_corr > 0);
        edges_neg = (edge_p < 0.05) & (edge_corr < 0);
    end

    all_pos_edges(:,i_fold) = edges_pos;
    all_neg_edges(:,i_fold) = edges_neg;

    % build model on TRAIN subs
    train_sum = (nansum(train.x(edges_pos, :), 1) - nansum(train.x(edges_neg, :), 1))';
    fit_train = polyfit(train_sum, train.y(:,1), 1);

    % run model on TEST sub
    test_sum = sum(test.x(edges_pos, :), 1) - sum(test.x(edges_neg,:), 1);

    pred_Y(test.indx) = (test_sum*fit_train(1)+fit_train(2))';

end

pred_Y = pred_Y';

%% Evaluate model performance
[stats.r_pearson, stats.p_pearson] = corr(pred_Y, y, 'rows', 'complete');
[stats.r_rank, stats.p_rank] = corr(pred_Y, y, 'type', 'spearman','rows','complete');
stats.mse = sum((pred_Y - y).^2) / N;
stats.q_s = 1 - stats.mse / var(y, 1,'omitnan');

end