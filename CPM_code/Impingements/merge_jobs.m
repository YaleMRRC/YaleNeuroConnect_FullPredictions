% Folder containing your .mat files
dataDir = '/Users/ajsimon/Documents/Data/Constable_lab/Transdiagnostic/N317/BSI_item_preds/indiv predictions/';   % <- CHANGE THIS
files = dir(fullfile(dataDir, '*.mat'));

% Preallocate (optional but cleaner)
r_preds       = [];
r_null_preds  = [];
p_values = [];
nets_both = [];
nets_pos = [];
nets_neg = [];

% r_non_overlap        = [];
% r_none   = [];

for i = 1:numel(files)
    % Load the file
    fname = fullfile(files(i).folder, files(i).name);
%     S = load(fname, 'r_multivar', 'r_multivar_null', 'r_overlap', 'r_overlap_null');
%     S = load(fname, 'r_none');
%     S = load(fname,'r_multivar', 'r_multivar_null');
    S = load(fname,'predictions', 'null_predictions','p_vals','networks');
    
%     % Concatenate 26x10 → eventually 26x100
%     r_multivar_all      = [r_multivar_all,      S.r_multivar];
%     r_multivar_null_all = [r_multivar_null_all, S.r_multivar_null];
    r_preds      = [r_preds,      S.predictions];
    r_null_preds = [r_null_preds, S.null_predictions];

    p_values = [p_values, (S.p_vals/1000)];

    x = eval(sprintf('S.networks.(''BSI item %d_both'')',i));

    nets_both = [nets_both,x];
    clear x

    x = eval(sprintf('S.networks.(''BSI item %d_pos'')',i));
    nets_pos = [nets_pos,x];
    clear x

    x = eval(sprintf('S.networks.(''BSI item %d_neg'')',i));
    nets_neg = [nets_neg,x];
    clear x
    
    % Concatenate 38x26x10 → eventually 38x26x100 (concat along 3rd dimension)
%     r_non_overlap       = cat(3, r_non_overlap,       S.r_non_overlap);
%     r_none  = cat(3, r_none,  S.r_none);
end

clearvars -except r*

save('/Users/ajsimon/Documents/Code/Symptom_cognitive_impingements/output/CPM_multivar_impingements_cogpreds.mat');

%%
% Folder containing your .mat files
dataDir = '/Users/ajsimon/Documents/Code/Symptom_cognitive_impingements/output/Symp_Splits/';   % <- CHANGE THIS
files = dir(fullfile(dataDir, '*.mat'));

% Preallocate empty (cell) to grow along first dim
overlap_networks_pos_all = [];
overlap_networks_neg_all = [];

for i = 1:numel(files)
    % Load only needed vars
    fname = fullfile(files(i).folder, files(i).name);
    S = load(fname, 'overlap_networks_pos', 'overlap_networks_neg');
    
    % Concatenate along dimension 1 (10 per file → 100 total)
    overlap_networks_pos_all = cat(1, overlap_networks_pos_all, S.overlap_networks_pos);
    overlap_networks_neg_all = cat(1, overlap_networks_neg_all, S.overlap_networks_neg);
end

% At this point both are:
% 100 x 10 x 38 x 26

% Now reshape to 1000 x 38 x 26
% (100 * 10 = 1000)
overlap_networks_pos_1000 = reshape(overlap_networks_pos_all, [10000, 38, 26]);
overlap_networks_neg_1000 = reshape(overlap_networks_neg_all, [10000, 38, 26]);

overlap_networks_pos = overlap_networks_pos_1000;
overlap_networks_neg = overlap_networks_neg_1000;

clearvars -except overlap_networks_neg overlap_networks_pos

thresh = 0.5;

for r = 1:38
    for c = 1:26

        pos_temp = zeros(35778,10000);
        neg_temp = zeros(35778,10000);

        for p = 1:10000
            these_posixs = overlap_networks_pos{p,r,c};
            these_posixs = double(these_posixs);

            these_posixs = these_posixs+1;

            these_negixs = overlap_networks_neg{p,r,c};
            these_negixs = double(these_negixs);

            these_negixs = these_negixs+1;

            pos_temp(these_posixs,p) = 1;
            neg_temp(these_negixs,p) = 1;

            clear these_posixs these_negixs 
            overlap_networks_pos{p,r,c} = [];
            overlap_networks_neg{p,r,c} = [];

        end

        pos_prop = mean(pos_temp, 2);                  % proportion of 1's in each row
        pos_mask(r,c,:) = double(pos_prop > thresh);

        neg_prop = mean(neg_temp, 2);                  % proportion of 1's in each row
        neg_mask(r,c,:) = double(neg_prop > thresh);

        clear pos_temp neg_temp pos_prop neg_prop

    end
end

clearvars -except pos_mask neg_mask

save('/Users/ajsimon/Documents/Code/Symptom_cognitive_impingements/output/CPM_impingement_SYMPTOM_prediction_masks.mat');


