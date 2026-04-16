% This script will run CPM for each symptom, compare those predictions to a
% null distribution, and calculate a network score by summing all edges
% that are predictive in the majority of CV folds

% to do:
% - save 2 separate header files for figures:
%   1. Has the assessment scale for each measure
%   2. Has the measure name in English

clear;

%% load stuff
load('/Users/ajsimon/Documents/Data/Constable_lab/Transdiagnostic/N317/CPM_input_data/averaged_mats.mat');
load('/Users/ajsimon/Documents/Data/Constable_lab/Transdiagnostic/N317/CPM_input_data/cogdata_to_model.mat');
load('/Users/ajsimon/Documents/Data/Constable_lab/Transdiagnostic/N317/CPM_input_data/confounds.mat');

%% Make sure these are correct each time you run this script!
outfile = '/Users/ajsimon/Documents/Data/Constable_lab/Transdiagnostic/N317/CPM_results/AgeSex_regressed_1000perms/cognitive_predictions.mat';

data = cogdata; clear cogdata
header = cogheader; clear cogheader

%% set parameters
nperms = 1000;
kfolds = 10;
edge_thresh = 0.5;
nedgethresh = edge_thresh*kfolds*nperms;

N = size(data,1);
nmeas = size(data,2);

%% Run predictions
for t = 1:nmeas

    fprintf('%s\n',header{1,t});
    feat_of_interest = cell2mat(data(:,t));

    ix=0;
    for i = 1:N
        if ~isnan(feat_of_interest(i,1))
            ix=ix+1;
            x(:,:,ix) = avg_mats(:,:,i);
            y(ix,1) = feat_of_interest(i,1);

            % format confounds
            Age(ix,1) = cell2mat(Confounds(i,1));
            
            ismale = strfind(Confounds{i,2},'M');
            if ~isempty(ismale)
                Sex(ix,1) = 1;
            else
                Sex(ix,1) = 2;
            end
        end
    end

    tic
    for np = 1:nperms

        if np == 1
            fprintf('Iteration ')
        end
        fprintf('%d ', np)

        % Run CPM
        [stats, all_pos_edges, all_neg_edges] = runCPM(x,y,kfolds,Age,Sex);

        Pred_strength(np,1) = stats.r_rank;

        ix1 = (np*10)-9;
        ix2 = np*10;
        pos_edges(:,ix1:ix2) = all_pos_edges;
        neg_edges(:,ix1:ix2) =  all_neg_edges;

        % Run  CPM on null model
        randY = randperm(numel(y));
        ShuffledY=reshape(y(randY),size(y));

        % Null = main(x,ShuffledY,kfolds,'results');

        [nullstats, ~, ~] = runCPM(x,ShuffledY,kfolds,Age,Sex);
        Null_preds(np,1) = nullstats.r_rank;

        clear ShuffledY randY nullstats ix1 ix2 all_pos_edges all_neg_edges stats
    end

    fprintf('\n');
    toc

    p_vals(t,1) = length(find(Null_preds > median(Pred_strength)));

    predictions(:,t) = Pred_strength;
    null_predictions(:,t) = Null_preds;

    % Compute network score
    all_pos_edges = sum(pos_edges,2);
    all_neg_edges = sum(neg_edges,2);

    pos_pred_edges = find(all_pos_edges > nedgethresh);
    neg_pred_edges = find(all_neg_edges > nedgethresh);

    pos_network = zeros(size(all_pos_edges));
    neg_network = zeros(size(all_neg_edges));

    pos_network(pos_pred_edges,1) = 1;
    neg_network(neg_pred_edges,1) = 1;

    eval(sprintf('networks.%s_pos = pos_network;',header{1,t}));
    eval(sprintf('networks.%s_neg = neg_network;',header{1,t}));
    eval(sprintf('networks.%s_both = pos_network + neg_network;',header{1,t}));

    % pull connectivity values from predictive edges
    for s = 1:N
        conn = squeeze(avg_mats(:,:,s));
        D = diag(conn);
        vec = [squareform((conn-diag(D)).')];
        netscores(s,t) = sum(vec(1,pos_pred_edges)) - sum(vec(1,neg_pred_edges));
        clear vec D conn
    end

    clear all_pos_edges all_neg_edges pos_pred_edges neg_pred_edges x y feat_of_interest ...
         pos_edges neg_edges Null_preds Pred_strength Sex Age pos_network neg_network

end

clearvars -except p_vals netscores predictions null_predictions networks

save(outfile);



