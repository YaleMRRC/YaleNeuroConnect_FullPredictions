#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import io as sio, stats
from scipy.spatial.distance import squareform
from scipy.stats import zscore
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

# ===================== Utilities =====================

def _vectorize_edges_stack(x3):
    n_nodes, _, n_subj = x3.shape
    n_edges = n_nodes * (n_nodes - 1) // 2
    out = np.empty((n_edges, n_subj), dtype=float)
    for s in range(n_subj):
        m = x3[:, :, s].copy()
        np.fill_diagonal(m, 0.0)
        out[:, s] = squareform(m, checks=False)
    return out

def fast_partial_corr_matrix(X, y, cov):
    X = np.asarray(X, float)
    y = np.asarray(y, float).flatten()
    cov = np.asarray(cov, float).flatten().reshape(-1, 1)

    mask = (~np.isnan(y)) & (~np.isnan(cov).any(axis=1)) & (~np.isnan(X).any(axis=0))
    X = X[:, mask]
    y = y[mask]
    cov = cov[mask]

    if X.shape[1] < 3:
        r = np.full(X.shape[0], np.nan)
        p = np.full(X.shape[0], np.nan)
        return r, p

    reg_y = LinearRegression().fit(cov, y)
    y_res = y - reg_y.predict(cov)

    reg_x = LinearRegression().fit(cov, X.T)
    X_res = X - reg_x.predict(cov).T

    y_c = y_res - y_res.mean()
    X_c = X_res - X_res.mean(axis=1, keepdims=True)

    num = np.sum(X_c * y_c, axis=1)
    den = np.sqrt(np.sum(X_c**2, axis=1)) * np.sqrt(np.sum(y_c**2))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    r = np.clip(r, -1.0, 1.0)

    n = X.shape[1]
    df = max(n - 2, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = r * np.sqrt(df / np.maximum(1e-12, 1 - r**2))
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=df))
    return r, p

def select_edges_with_cov(train_edges, train_y, train_age, train_sex, alpha=0.05):
    r_age, p_age = fast_partial_corr_matrix(train_edges, train_y, train_age)
    r_sex, p_sex = fast_partial_corr_matrix(train_edges, train_y, train_sex)
    edges_pos = (p_age < alpha) & (r_age > 0) & (p_sex < alpha) & (r_sex > 0)
    edges_neg = (p_age < alpha) & (r_age < 0) & (p_sex < alpha) & (r_sex < 0)
    return edges_pos, edges_neg

def compute_overlap_masks(clinical_pos, clinical_neg, cog_pos, cog_neg):
    clinical_pos = np.asarray(clinical_pos, bool).reshape(-1)
    clinical_neg = np.asarray(clinical_neg, bool).reshape(-1)
    cog_pos      = np.asarray(cog_pos,      bool).reshape(-1)
    cog_neg      = np.asarray(cog_neg,      bool).reshape(-1)
    pos_in_overlap = clinical_pos & cog_pos
    neg_in_overlap = clinical_neg & cog_neg
    return pos_in_overlap, neg_in_overlap

def compute_overlap_weights(
    cog_pos, cog_neg,Weighted_CPMnets_vecs_all):
    
    # --- Convert cog network masks to booleans ---
    cog_pos      = np.asarray(cog_pos,      dtype=bool).reshape(-1, 1)  # (35778, 1)
    cog_neg      = np.asarray(cog_neg,      dtype=bool).reshape(-1, 1)

    # Weights
    weights = np.asarray(Weighted_CPMnets_vecs_all.T, dtype=float)  #

    # --- Project deviance edges onto masks ----
    pos_impingements = (weights * cog_pos)
    neg_impingements = -(weights * cog_neg)   # invert to get on same scale as positive deviances
    all_impingements = pos_impingements + neg_impingements

    # --- Per-subject sums (sum of deviance values where mask == 1) ---
    sums = {
        "pos_impingements":         (pos_impingements).sum(axis=0),
        "neg_impingements":              (neg_impingements).sum(axis=0),
        "all_impingements":          (all_impingements).sum(axis=0),
    }

    return sums

def corr_no_nan(x, y, kind="spearman"):
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    m = (~np.isnan(x)) & (~np.isnan(y))
    if m.sum() < 3:
        return np.nan, np.nan
    if kind == "pearson":
        return stats.pearsonr(x[m], y[m])
    else:
        return stats.spearmanr(x[m], y[m])

def load_array_from_path(path):
    path = str(path)
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
        header = df.columns.to_numpy()
        arr = df.to_numpy(dtype=float)
        return arr, header
    else:
        md = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        for key in ["cogdata", "sympdata", "avg_mats", "Confounds"]:
            if key in md:
                return md[key], None
        for k, v in md.items():
            if not k.startswith("__"):
                return v, None
        raise ValueError(f"Could not find array in {path}")


# ===================== Core Overlap-CPM =====================

def run_overlap_cpm(
    avg_mats,           # (Nnodes, Nnodes, Nsubjects)
    cogdata,            # (Nsubjects, Ncog)
    sympdata,           # (Nsubjects, Nsymp)
    Age, Sex,           # (Nsubjects,), numeric sex 1/2
    kfolds=10,
    rng=None
):

    if rng is None:
        rng = np.random.default_rng()
        
    Nnodes, _, N = avg_mats.shape
    Ncog = cogdata.shape[1]
    Nsymp = sympdata.shape[1]

    # Vectorize edges once
    all_edges = _vectorize_edges_stack(avg_mats)  # (Nedges, N)
    Nedges = all_edges.shape[0]

    all_edges_z = zscore(all_edges, axis=1, ddof=0, nan_policy='omit')

    deviances = np.asarray(all_edges_z, dtype=np.float32)[:, :]  # (35778, 317, 1)

    # Outputs
    yhat_overlap = np.full((N, Nsymp, Ncog), np.nan)     # predictions per subject for each (symp, cog)
    yhat_overlap_null = np.full((N, Nsymp, Ncog), np.nan)     # predictions per subject for each (symp, cog)

    yhat_multivar = np.full((N, Ncog), np.nan)     # predictions per subject for each
    yhat_multivar_null = np.full((N, Ncog), np.nan)   

    overlap_pos_idx = [ [ [None for _ in range(Ncog)] for _ in range(Nsymp)] for _ in range(kfolds) ]
    overlap_neg_idx = [ [ [None for _ in range(Ncog)] for _ in range(Nsymp)] for _ in range(kfolds) ]

    r_multivar   = np.full((Ncog), np.nan)
    p_multivar   = np.full((Ncog), np.nan)
    r_multivar_null   = np.full((Ncog), np.nan)
    p_multivar_null   = np.full((Ncog), np.nan)

    r_overlap   = np.full((Nsymp, Ncog), np.nan)
    p_overlap   = np.full((Nsymp, Ncog), np.nan)
    r_overlap_null   = np.full((Nsymp, Ncog), np.nan)
    p_overlap_null   = np.full((Nsymp, Ncog), np.nan)

    kf = KFold(n_splits=kfolds, shuffle=True)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(np.arange(N))):

        # define train and test
        train_edges = all_edges[:, train_idx]   # edges x Ntrain
        test_edges  = all_edges[:, test_idx]    # edges x Ntest

        train_devs = deviances[:, train_idx]   # edges x Ntrain
        test_devs  = deviances[:, test_idx]    # edges x Ntest

        train_age = Age[train_idx]
        train_sex = Sex[train_idx]

        perm = rng.permutation(cogdata.shape[0])
        shuffled_cog = cogdata[perm, :]

        perm = rng.permutation(sympdata.shape[0])
        shuffled_symp = sympdata[perm, :]
        
        # --- Train symptoms ---
        symp_pos_masks = np.zeros((Nsymp, Nedges), dtype=bool)
        symp_neg_masks = np.zeros((Nsymp, Nedges), dtype=bool)

        for s in range(Nsymp):
            y_s = sympdata[train_idx, s]
            pos_s, neg_s = select_edges_with_cov(train_edges, y_s, train_age, train_sex)
            symp_pos_masks[s, :] = pos_s
            symp_neg_masks[s, :] = neg_s

        # --- Train cognition ---
        cog_pos_masks = np.zeros((Ncog, Nedges), dtype=bool)
        cog_neg_masks = np.zeros((Ncog, Nedges), dtype=bool)

        for c in range(Ncog):
            y_c = cogdata[train_idx, c]
            pos_c, neg_c = select_edges_with_cov(train_edges, y_c, train_age, train_sex)
            cog_pos_masks[c, :] = pos_c
            cog_neg_masks[c, :] = neg_c

        ####### Use the multivariate weighting method to predict cognition ######
        trainsymps = sympdata[train_idx, :]
        traincogs = cogdata[train_idx, :]

        traincogs_shuffled = shuffled_cog[train_idx, :]

        testsymps = sympdata[test_idx, :]
        testcogs = cogdata[test_idx, :]

        # Compute multivariate-weighted CPM networks
        Weighted_CPMnets_vecs_pos = np.empty((sympdata.shape[0],symp_pos_masks.shape[1]))
        Weighted_CPMnets_vecs_neg = np.empty((sympdata.shape[0],symp_neg_masks.shape[1]))

        for n in range(sympdata.shape[0]):   #n training subs

            Weighted_vecs_pos_temp = np.empty((symp_pos_masks.shape[0], symp_pos_masks.shape[1]))
            Weighted_vecs_neg_temp = np.empty((symp_neg_masks.shape[0], symp_pos_masks.shape[1]))

            for s in range(sympdata.shape[1]):   # n symptoms
                w = sympdata[n, s]

                Weighted_vecs_pos_temp[s,:] = symp_pos_masks[s,:] * w
                Weighted_vecs_neg_temp[s,:] = symp_neg_masks[s,:] * w

            Weighted_CPMnets_vecs_pos[n,:] = np.nansum(Weighted_vecs_pos_temp, axis=0)
            Weighted_CPMnets_vecs_neg[n,:] = np.nansum(Weighted_vecs_neg_temp, axis=0)

        Weighted_CPMnets_vecs_neg = Weighted_CPMnets_vecs_neg*-1
        Weighted_CPMnets_vecs_all = Weighted_CPMnets_vecs_pos + Weighted_CPMnets_vecs_neg

        sum_weights_impingements = np.zeros((Ncog,sympdata.shape[0]))

        for c in range(Ncog):
            cog_pos = cog_pos_masks[c, :]
            cog_neg = cog_neg_masks[c, :]

            sums = compute_overlap_weights(cog_pos, cog_neg, Weighted_CPMnets_vecs_all)

            # index overlap info for each subject
            for s in range(sympdata.shape[0]):
                sum_weights_impingements[c,s] = sums["all_impingements"][s]

        for co in range(26):   # Specify number of cognitive measures

            # build linear model between weighted multivariate CPM matrix and cognitive measures
            x = sum_weights_impingements[co, train_idx]
            y = traincogs[:,co]

            mask = ~np.isnan(x) & ~np.isnan(y)   # keep only valid values
            a, b = np.polyfit(x[mask], y[mask], 1)

            # Predict on TEST
            yhat_multivar[test_idx, co] = a * sum_weights_impingements[co,test_idx] + b

            # Perform null testing
            y = traincogs_shuffled[:,co]

            mask = ~np.isnan(x) & ~np.isnan(y)   # keep only valid values
            a, b = np.polyfit(x[mask], y[mask], 1)

            yhat_multivar_null[test_idx, co] = a * sum_weights_impingements[co,test_idx] + b


        #### Predict cognition from the overlapping edges for each clinical-cognitive network pair
        for s in range(Nsymp):
            clin_pos = symp_pos_masks[s, :]
            clin_neg = symp_neg_masks[s, :]

            for c in range(Ncog):
                cog_pos = cog_pos_masks[c, :]
                cog_neg = cog_neg_masks[c, :]

                # Store overlapping edge masks
                pos_ov, neg_ov = compute_overlap_masks(clin_pos, clin_neg, cog_pos, cog_neg)

                pos_indices = np.where(pos_ov)[0]
                neg_indices = np.where(neg_ov)[0]

                overlap_pos_idx[fold_idx][s][c] = pos_indices
                overlap_neg_idx[fold_idx][s][c] = neg_indices

                if pos_indices.size == 0 and neg_indices.size == 0:
                    continue

                # Build training summary for the cognitive target c
                train_sum = np.nansum(train_devs[pos_ov, :], axis=0) - (np.nansum(train_devs[neg_ov, :], axis=0))
                y_train_c = cogdata[train_idx, c]

                m = (~np.isnan(train_sum)) & (~np.isnan(y_train_c))

                a, b = np.polyfit(train_sum[m], y_train_c[m], 1)

                # Predict on TEST
                test_sum = np.nansum(test_devs[pos_ov, :], axis=0) - (np.nansum(test_devs[neg_ov, :], axis=0))
                yhat_overlap[test_idx, s, c] = a * test_sum + b

                # Perform null tests
                y_train_c_null = shuffled_cog[train_idx, c]

                m = (~np.isnan(train_sum)) & (~np.isnan(y_train_c_null))

                a, b = np.polyfit(train_sum[m], y_train_c_null[m], 1)

                # Predict on TEST
                test_sum = np.nansum(test_devs[pos_ov, :], axis=0) - (np.sum(test_devs[neg_ov, :], axis=0))
                yhat_overlap_null[test_idx, s, c] = a * test_sum + b

    # --- Evaluate performance across all subjects ---
    ###### Multivariately weighted method
    for c in range(Ncog):
        r_multivar[c],   p_multivar[c]   = corr_no_nan(yhat_multivar[:, c], cogdata[:, c], kind="spearman")

    # Null testing
    for c in range(Ncog):
        r_multivar_null[c],   p_multivar_null[c]   = corr_no_nan(yhat_multivar_null[:, c], cogdata[:, c], kind="spearman")

    ###### Overlap method

    for s in range(Nsymp):
        for c in range(Ncog):
            r_overlap[s, c],   p_overlap[s, c]   = corr_no_nan(yhat_overlap[:, s, c], cogdata[:, c], kind="spearman")

    # Null testsing
    for s in range(Nsymp):
        for c in range(Ncog):
            r_overlap_null[s, c],   p_overlap_null[s, c]   = corr_no_nan(yhat_overlap_null[:, s, c], cogdata[:, c], kind="spearman")

    return {
        'predicted_cog_overlap': yhat_overlap,    
        'predicted_cog_multivar': yhat_multivar, 
        'r_multivar': r_multivar, 
        'p_multivar': p_multivar,
        'r_multivar_null': r_multivar_null, 
        'r_overlap': r_overlap, 
        'p_overlap': p_overlap,
        'r_overlap_null': r_overlap_null, 
        'overlap_networks_pos': overlap_pos_idx,
        'overlap_networks_neg': overlap_neg_idx,
    }

def parse_args():
    p = argparse.ArgumentParser(description="Overlap-CPM (HPC-ready)")
    p.add_argument("--job-id", type=int, required=True, help="SLURM_ARRAY_TASK_ID (1..10)")
    p.add_argument("--perm-count", type=int, default=10, help="Permutations to run in THIS job")
    p.add_argument("--outfile-dir", type=str, default="/home/ajs332/project/Impingement_predictions/results/")
    p.add_argument("--averaged-mats", type=str, default="/home/ajs332/project/Impingement_predictions/averaged_mats.mat")
    p.add_argument("--cogdata", type=str, default="/home/ajs332/project/Impingement_predictions/cogdata_normalized.csv")
    p.add_argument("--sympdata", type=str, default="/home/ajs332/project/Impingement_predictions/sympdata_normalized.csv")
    p.add_argument("--confounds", type=str, default="/home/ajs332/project/Impingement_predictions/confounds.mat")
    p.add_argument("--kfolds", type=int, default=10)
    p.add_argument("--base-seed", type=int, default=4242, help="Base seed; each perm = base + global_perm_index")
    return p.parse_args()

def make_rngs(job_id:int, perm_count:int, base_seed:int):
    rngs = []
    for p in range(perm_count):
        global_perm_index = (job_id - 1) * perm_count + p
        seed = base_seed + global_perm_index
        rngs.append(np.random.default_rng(seed))
    return rngs

def main():
    args = parse_args()

    outdir = Path(args.outfile_dir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    outfile_mat = outdir / f"cpm_overlap_predictions_job{args.job_id}.mat"
    outfile_npz = outdir / f"cpm_overlap_predictions_job{args.job_id}.npz"

    mats = sio.loadmat(args.averaged_mats, squeeze_me=True, struct_as_record=False)
    avg_mats = mats["avg_mats"]

    cogdata_arr, cogheader = load_array_from_path(args.cogdata)
    sympdata_arr, sympheader = load_array_from_path(args.sympdata)
    if cogdata_arr.ndim == 1: cogdata_arr = cogdata_arr.reshape(-1, 1)
    if sympdata_arr.ndim == 1: sympdata_arr = sympdata_arr.reshape(-1, 1)

    N = cogdata_arr.shape[0]
    conf = sio.loadmat(args.confounds, squeeze_me=True, struct_as_record=False)
    Confounds = conf["Confounds"]
    Age = np.zeros(N, float)
    Sex = np.zeros(N, float)
    for i in range(N):
        age_val = Confounds[i, 0]
        try:
            Age[i] = float(age_val) if np.size(age_val) == 1 else float(np.ravel(age_val)[0])
        except Exception:
            Age[i] = np.nan
        Sex[i] = 1.0 if ('M' in str(Confounds[i, 1])) or ('m' in str(Confounds[i, 1])) else 2.0

            
    rngs = make_rngs(args.job_id, args.perm_count, args.base_seed)

    results_perms = []
    for rng in rngs:
        res = run_overlap_cpm(
            avg_mats=avg_mats,
            cogdata=cogdata_arr,
            sympdata=sympdata_arr,
            Age=Age, Sex=Sex,
            kfolds=args.kfolds,
            rng=None,
        )
        results_perms.append(res)

    outdict = {}
    for key in results_perms[0].keys():
        vals = [res[key] for res in results_perms]
        if isinstance(vals[0], np.ndarray):
            outdict[key] = np.stack(vals, axis=-1)  # (..., nperms)
        else:
            outdict[key] = vals  # lists of indices per perm

    if cogheader is not None: outdict["cogheader"] = cogheader
    if sympheader is not None: outdict["sympheader"] = sympheader

    sio.savemat(str(outfile_mat), outdict)
    np.savez(str(outfile_npz), **outdict)
    print(f"Saved:\n  {outfile_mat}\n  {outfile_npz}")

if __name__ == "__main__":
    main()
