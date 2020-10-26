import pandas as pd
import numpy as np
import itertools
import re
import time
import joblib
import copy
import shutil
import os
import pprint as pp
from collections import namedtuple
import argparse
import gc
import glob
from tqdm.auto import tqdm

import util
from util import mprof_timestamp, str2bool
from models import LgbmModel, CatBoostModel, XgbModel, RidgeModel, MLPModel, LogisticModel, RFModel, MeanModel
from preproc import create_folds

import logging_config
from logging import getLogger
logger = getLogger("splt")

pd.options.display.max_rows = 1000

SelectedConfig = namedtuple("SelectedConfig", "cols files cols_exc files_stacking")


def shape(df):
    if df is None:
        return None
    else:
        return df.shape


def load_1file(file_path, rows, selected_idx):
    d = joblib.load(file_path)
    if selected_idx is not None:
        d = d.loc[selected_idx,:]
    if rows is not None:
        #d = d.iloc[:rows]
        d = d.sample(n=rows, random_state=0).sort_index().reset_index(drop=True)
    return d


def comp_set(s1, s2):
    res = f"{len(s1)} {len(s2)}\ns1-s2 = {s1-s2}\ns2-s1 = {s2-s1}"
    return res


def reduce_memory(df):
    df = df.astype(np.float32)
    return df


def select_cols(df, selected_cols, excluded_cols):
    cols = df.columns
    if selected_cols:
        cols = [c for c in cols if c in selected_cols]
    if excluded_cols:
        cols = [c for c in cols if c not in excluded_cols]
    if len(cols) < len(df.columns):
        df = df[cols]
    return df     


def is_selected(x, selected_files):
    if selected_files is None:
        return True
    else:
        #return x in selected_files  # 完全一致
        mts = [s for s in selected_files if re.match(s, x)]  # 前方一致
        return len(mts) > 0


def load_selected_idx(features_path :str, restrict_lobby_mode :str):
    if restrict_lobby_mode is None:
        return None
    else:
        merge = joblib.load(features_path + "/merge.joblib")
        restrict_idx = merge.index[merge["lobby-mode"] == restrict_lobby_mode]
        return restrict_idx


def load_and_preproc_stacking(features_path, selected_config :SelectedConfig, args):
    rows, selected_idx = None, None

    def _load_1file(file_path):
        return load_1file(file_path, rows, selected_idx)

    def _is_selected(x):
        return is_selected(x, selected_config.files_stacking)

    merge = _load_1file(features_path + "/merge.joblib")
    if args.fold_num_stacking is not None:
        logger.info("Recreate folds for stacking.")
        merge["fold"] = create_folds(merge, args.fold_num_stacking, args.shuffle_folds_stacking, args.stratified_folds_stacking, args.seed_folds_stacking)
    merge = merge[["split_type", "fold", "y"]]

    ps = sorted(glob.glob("../models/20*/preds_valid.csv"))
    if not args.include_stacking_preds:
        ps = [p for p in ps if re.search("stacking", p) is None]
    ps = [p for p in ps if _is_selected(p.split("/")[-2])]
    ps2 = []

    preds = []
    for p in ps:
        m = pd.read_csv(p)["pred"]
        if (m.shape[0] != 66125) or (m.isnull().any()):
            logger.info(f"Pass preds_valid {p}")
            continue
        m.name = p.split("/")[-2]
        preds.append(m)
        ps2.append(p)

    preds_valid = pd.concat(preds, axis=1)
    assert not preds_valid.isnull().any().any()

    preds = []
    for p in ps2:
        m = pd.read_csv(os.path.dirname(p) + "/preds_test.csv")
        check = m.shape[0]!=28340 or m.shape[1]!=11 or m.isnull().any().any()
        assert check == False, p
        m = m.iloc[:,1:].mean(axis=1)
        m.name = p.split("/")[-2]
        preds.append(m)
    
    preds_test = pd.concat(preds, axis=1)
    assert not preds_test.isnull().any().any()

    assert all(preds_valid.columns == preds_test.columns)

    preds_merge = pd.concat([preds_valid, preds_test], axis=0).reset_index(drop=True)

    merge = pd.concat([merge, preds_merge], axis=1)

    return merge
    

def load_and_preproc_common(features_path, selected_config :SelectedConfig, do_stage_int :bool, rows, selected_idx, args):
    selected_cols = selected_config.cols
    selected_files = selected_config.files
    excluded_cols = selected_config.cols_exc

    def _load_1file(file_path):
        return load_1file(file_path, rows, selected_idx)

    def _select_cols(df):
        return select_cols(df, selected_cols, excluded_cols)

    def _is_selected(x):
        return os.path.exists(f"{features_path}/{x}") and is_selected(x, selected_files)

    ms = []
    load_base_file_N = 0

    if _is_selected("merge.joblib"):
        merge = _load_1file(features_path + "/merge.joblib")

        if do_stage_int:
            merge["stage"] = merge["stage"].cat.codes
        
        merge.drop(["id", "period"], axis=1, inplace=True)
        ms.append(merge)
        del merge
        load_base_file_N += 1

    if _is_selected("merge_tm.joblib"):
        merge_tm = _load_1file(features_path + "/merge_tm.joblib")
        ms.append(merge_tm)
        del merge_tm
        load_base_file_N += 1

    if _is_selected("merge_agg.joblib"):
        merge_agg = _load_1file(features_path + "/merge_agg.joblib")
        ms.append(merge_agg)
        del merge_agg
        load_base_file_N += 1

    if _is_selected("merge_agg-v15.joblib"):
        merge_agg = _load_1file(features_path + "/merge_agg-v15.joblib")
        ms.append(merge_agg)
        del merge_agg
        load_base_file_N += 1

    if _is_selected("merge_wc.joblib"):
        merge_wc = _load_1file(features_path + "/merge_wc.joblib")
        ms.append(merge_wc)
        del merge_wc
        load_base_file_N += 1

    ms = [_select_cols(m) for m in ms]
    
    cs = set(itertools.chain.from_iterable([m.columns for m in ms]))
    ps = sorted(glob.glob(features_path + f"/merge_freqenc*.joblib"))
    if args.features_path_prev:
        ps = [args.features_path_prev + "/merge_freqenc.joblib"] + ps
    logger.info(f"len(ps)={len(ps)}")
    for p in ps:
        if not _is_selected(os.path.basename(p)):
            logger.info(f"Pass common {p}")
            continue
        
        logger.info(f"Loading common {p}")
        m = _load_1file(p)
        cs_rmv = [c for c in m.columns if c in cs]
        if len(cs_rmv):
            logger.warning(f"Remove duplicated columns path={p} cols[:5]={cs_rmv}[:5]")
            m.drop(cs_rmv, axis=1, inplace=True)

        m = _select_cols(m)

        ms.append(m)
        cs = cs | set(m.columns)
        
    logger.info(f"Loaded common load_N={len(ms)} pass_N={(len(ps)+4)-len(ms)}")

    if args.do_save_data and (len(ms) > load_base_file_N):
        logger.info("Saving freqenc data.")
        st = time.time()
        ms2 = pd.concat(ms[load_base_file_N:], axis=1) 
        joblib.dump(ms2, f"{args.models_path}/merge_freqenc.joblib", compress=4)
        del ms2
        logger.info(f"Saved freqenc data. {(time.time()-st):.0f}sec")
    
    ms = pd.concat(ms, axis=1)

    return ms


def load_and_preproc_1fold(features_path, fold, selected_config :SelectedConfig, do_reduce_memory, rows, selected_idx, args):
    def _load_1file(file_path):
        return load_1file(file_path, rows, selected_idx)

    ps = sorted(glob.glob(features_path + f"/*-f{fold:02d}.joblib"))
    if args.features_path_prev:
        ps = [args.features_path_prev + f"/merge_tarenc-f{fold:02d}.joblib"] + ps

    ms = []
    cs = set()
    for p in tqdm(ps):
        if not is_selected(os.path.basename(p), selected_config.files):
            if fold == 0:
                logger.info(f"Pass f{fold:02d} {p}")
            continue

        if fold == 0:
            logger.info(f"Loading f{fold:02d} {p}")

        m = _load_1file(p)
        if do_reduce_memory:
            m = reduce_memory(m)
        cs_rmv = [c for c in m.columns if c in cs]
        if len(cs_rmv):
            logger.warning(f"Remove duplicated columns path={p} cols[:5]={cs_rmv[:5]}")
            m.drop(cs_rmv, axis=1, inplace=True)

        m = select_cols(m, selected_config.cols, selected_config.cols_exc)

        ms.append(m)
        cs = cs | set(m.columns)
        del m
        gc.collect()

    logger.info(f"Loaded f{fold:02d} load_N={len(ms)} pass_N={len(ps)-len(ms)}")

    if len(ms) > 0:
        ms = pd.concat(ms, axis=1) 
        assert ms.columns.duplicated().any() == False
    else:
        ms = None

    if args.do_save_data:
        logger.info(f"Saving tarenc data f{fold:02d}.")
        st = time.time()
        joblib.dump(ms, f"{args.models_path}/merge_tarenc-f{fold:02d}.joblib", compress=4)  # {os.path.basename(args.models_path)}
        logger.info(f"Saved tarenc data f{fold:02d}. {(time.time()-st):.0f}sec")

    return ms


def init_model_dir(_args):
    args = copy.deepcopy(_args)

    timestr = util.get_time_str()
    args.models_path = "../models/" + timestr + "-" + args.suffix

    if args.DO_TEST:
        args.models_path = args.test_models_path
        util.trash(args.models_path)
    else:
        util.trash(args.models_path)

    os.makedirs(args.models_path, exist_ok=False)

    logging_config.init(f"{args.models_path}/log_{timestr}.log")

    util.dump_json(vars(args), f"{args.models_path}/args.json")
    logger.info("args after init_model_dir =\n" + pp.pformat(vars(args)))

    shutil.copytree(args.src_path, args.models_path + "/src_" + timestr)

    return args


def load_selected_config(selected_config_path):
    scs = util.load_json(selected_config_path)

    if "selected_cols" in scs:
        assert len(set(scs["selected_cols"]) & {"y", "split_type", "fold"}) == 3, set(scs["selected_cols"]) & {"y", "split_type", "fold"}
        sc = scs["selected_cols"]
        logger.info(f"selected_cols_N={len(sc)}")
    else:
        sc = None

    if "excluded_cols" in scs:
        ec = set(scs["excluded_cols"])
        logger.info(f"excluded_cols_N={len(ec)}")
    else:
        ec = None
    
    if "selected_files" in scs:
        sf = scs["selected_files"]
        logger.info(f"selected_files_N={len(sf)}")
    else:
        sf = None

    if "selected_files_stacking" in scs:
        sf_st = scs["selected_files_stacking"]
        logger.info(f"selected_files_stacking_N={len(sf_st)}")
    else:
        sf_st = None

    selected_config = SelectedConfig(sc, sf, ec, sf_st)
    return selected_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--DO_TEST', action='store_true')
    parser.add_argument('--features_path', type=str, required=True)
    parser.add_argument('--features_path_prev', type=str, default=None)
    parser.add_argument('--selected_config_path', type=str, default=None)
    parser.add_argument('--src_path', type=str, default="../src")
    parser.add_argument('--suffix', type=str, required=True)
    parser.add_argument('--test_models_path', type=str, default="../models/test")
    parser.add_argument('--train_folds', type=str, default=None)

    parser.add_argument('--do_train', type=str2bool, default=True)
    parser.add_argument('--do_save_data', action='store_true')
    parser.add_argument('--do_stacking', action='store_true')

    parser.add_argument('--early_stopping_metrics', type=str, choices=["accuracy", "logloss"], default="logloss")

    parser.add_argument('--do_stage_int', action='store_true')

    parser.add_argument('--params_lgbm_path', type=str, default=None)

    parser.add_argument('--model_type', type=str, choices=["lgbm", "catboost", "xgb", "ridge", "mlp", "logistic", "rf", "mean"], default="lgbm")

    parser.add_argument('--do_reduce_memory', action='store_true')

    parser.add_argument('--num_iterations', type=int, default=1000)

    parser.add_argument('--selected_lobby_mode', type=str, choices=["regular", "gachi"], default=None)

    parser.add_argument('--fold_num_stacking', type=int, default=None)
    parser.add_argument('--shuffle_folds_stacking', type=str2bool, default=None)
    parser.add_argument('--stratified_folds_stacking', type=str2bool, default=None)
    parser.add_argument('--seed_folds_stacking', type=int, default=None)
    parser.add_argument('--include_stacking_preds', action='store_true')

    parser.add_argument('--train_seed', type=int, default=100)
    parser.add_argument('--rf_min_samples_leaf', type=int, default=1)
    parser.add_argument('--rf_min_samples_split', type=int, default=2)
    parser.add_argument('--rf_n_estimators', type=int, default=200)
    parser.add_argument('--rf_max_depth', type=int, default=10)

    parser.add_argument('--cb_depth', type=int, default=None)
    parser.add_argument('--cb_rsm', type=int, default=None)
    parser.add_argument('--cb_min_data_in_leaf', type=int, default=None)

    parser.add_argument('--lg_num_leaves', type=int, default=None)
    parser.add_argument('--lg_min_data_in_leaf', type=int, default=None)
    parser.add_argument('--lg_lambda_l2', type=float, default=None)
    parser.add_argument('--lg_max_depth', type=int, default=None)
    parser.add_argument('--lg_feature_fraction', type=float, default=None)
    parser.add_argument('--lg_bagging_freq', type=int, default=None)
    parser.add_argument('--lg_bagging_fraction', type=float, default=None)

    args = parser.parse_args()

    args.rows = None

    if args.DO_TEST:
        #args.features_path = "../features/test"
        args.train_folds = "0"
        args.rows = 1000

    # LGBMのパラメータを設定
    params = dict()
    params_custom = dict()

    params["lgbm"] = {
        'objective': 'binary',
        'metric': 'None', 
        'num_iterations' : args.num_iterations,  # 最大イテレーション回数    
        'early_stopping_rounds' : 100,  # early_stopping 回数
        'first_metric_only': True,  # early stoppingで考慮するmetrics
        'seed': args.train_seed
    }

    params_custom["lgbm"] = {
        "early_stopping_metrics": args.early_stopping_metrics
    }

    if args.lg_num_leaves:
        params["lgbm"]["num_leaves"] = args.lg_num_leaves

    if args.lg_min_data_in_leaf:
        params["lgbm"]["min_data_in_leaf"] = args.lg_min_data_in_leaf

    if args.lg_lambda_l2:
        params["lgbm"]["lambda_l2"] = args.lg_lambda_l2

    if args.lg_max_depth:
        params["lgbm"]["max_depth"] = args.lg_max_depth

    if args.lg_feature_fraction:
        params["lgbm"]["feature_fraction"] = args.lg_feature_fraction

    if args.lg_bagging_freq:
        params["lgbm"]["bagging_freq"] = args.lg_bagging_freq

    if args.lg_bagging_fraction:
        params["lgbm"]["bagging_fraction"] = args.lg_bagging_fraction

    params["catboost"] = {
        "custom_loss": ['Accuracy'],
        "random_seed": args.train_seed,
        "logging_level": 'Verbose',
        #'verbose_eval': 10,
        "early_stopping_rounds": 100,
        "use_best_model": True,  # TrueだとBestIter以降のtreeは保存されない
        #train_dir="../tmp",
        #plot=True
    }

    if args.cb_depth:
        params["catboost"]["depth"] = args.cb_depth

    if args.cb_rsm:
        params["catboost"]["rsm"] = args.cb_rsm
    
    if args.cb_min_data_in_leaf:
        params["catboost"]["min_data_in_leaf"] = args.cb_min_data_in_leaf

    params_custom["catboost"] = params_custom["lgbm"]

    params["xgb"] = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'seed': args.train_seed
    }

    params_custom["xgb"] = {
        'early_stopping_rounds': 100,
        'num_boost_round': args.num_iterations,
    }

    params["ridge"] = {
        'random_state': args.train_seed
    }

    params_custom["ridge"] = {
    }

    params["logistic"] = {
        'random_state': args.train_seed,
        'max_iter': 1000,
    #    'C': 0.001,
    #    'penalty': 'none', # l2 l1
    #    'solver' : 'liblinear',
    }

    params_custom["logistic"] = {
    }
    
    params["mlp"] = {
        'batch_size': 64,
        'layer_num': 8,
        'drop_out_rate': 0.2
    }

    params_custom["mlp"] = {
    }

    params["rf"] = {
        'random_state': args.train_seed,
        'min_samples_leaf': args.rf_min_samples_leaf,  # 1
        'min_samples_split': args.rf_min_samples_split,  # 2
        'n_estimators': args.rf_n_estimators,  # 200
        'max_depth': args.rf_max_depth,  # 10
        #'oob_score': True,
        #'max_samples': 0.8,
        #'bootstrap': False,
    }

    params_custom["rf"] = {
    }

    params["mean"] = {
    }

    params_custom["mean"] = {
    }

    if args.early_stopping_metrics == "accuracy":
        params["catboost"]["eval_metric"] = "Accuracy"

    model2class = {"lgbm":LgbmModel, "catboost":CatBoostModel, 'xgb':XgbModel, 'ridge':RidgeModel, 'mlp':MLPModel, 'logistic':LogisticModel, 'rf':RFModel, 'mean':MeanModel}
    model_class = model2class[args.model_type]

    args.params = params[args.model_type]
    args.params_custom = params_custom[args.model_type]

    if args.train_folds:
        args.train_folds = [int(x) for x in args.train_folds.split(",")]

    args = init_model_dir(args)

    if args.selected_config_path:
        selected_config = load_selected_config(args.selected_config_path)
    else:
        selected_config = SelectedConfig(None, None, None, None)

    mprof_timestamp("load_preproc_common")
    logger.info("load_preproc_common")
    selected_idx = load_selected_idx(args.features_path, args.selected_lobby_mode)
    if selected_idx is None:
        logger.info(f"selected_idx = None")
    else:
        logger.info(f"selected_idx_len = {len(selected_idx)}")

    if args.do_stacking:
        merge_xy = load_and_preproc_stacking(args.features_path, selected_config, args)
    else:
        merge_xy = load_and_preproc_common(args.features_path, selected_config, args.do_stage_int, args.rows, selected_idx, args)
    cols_common = list(merge_xy.columns)
    logger.info(f"merge_xy.shape = {merge_xy.shape}")
    merge_xy.dtypes.to_csv(args.models_path + "/dtypes_common_1.csv")

    if args.params_lgbm_path:
        params_new = util.load_json(args.params_lgbm_path)["params"]
        for p in ["lambda_l1", "lambda_l2", "num_leaves", "feature_fraction", "bagging_fraction", "bagging_freq", "min_child_samples"]:
            args.params[p] = params_new[p]
        logger.info("Loaded params_lgbm =\n" + pp.pformat(args.params_lgbm))

    is_train_folds_full = False
    if args.train_folds is None:
        folds_num = merge_xy["fold"].max() + 1
        args.train_folds = range(folds_num)
        is_train_folds_full = True

    logger.info(f"Building Model.")
    model = model_class(merge_xy, args.models_path)
    model.train_x_common.dtypes.to_csv(args.models_path + "/dtypes_common_2.csv")
    del merge_xy
    gc.collect()

    for fold in args.train_folds:
        if args.do_stacking:
            mf = None
        else:
            mprof_timestamp(f"load_preproc_f{fold:02d}")
            logger.info(f"load_preproc_f{fold:02d}")
            mf = load_and_preproc_1fold(args.features_path, fold, selected_config, args.do_reduce_memory, args.rows, selected_idx, args)
            if fold == 0:
                if mf is not None:
                    mf.dtypes.to_csv(args.models_path + "/dtypes_fold.csv")
            logger.info(f"mf.shape = {shape(mf)} ")
            
            if selected_config.cols:
                cs1 = set(cols_common) | set(mf.columns) | {"y", "split_type", "fold"}  # 各model独自のpreprocで除かれる前で判定する
                cs2 = set(selected_config.cols)
                assert cs1 == cs2, comp_set(cs1, cs2)

        model.merge_1fold = mf
        if selected_config.cols is None:
            model.selected_cols = selected_config.cols
        else:
            model.selected_cols = [c for c in selected_config.cols if c not in {"y", "split_type", "fold"}]  # 列順keepのため
        del mf
        gc.collect()

        mprof_timestamp(f"train_f{fold:02d}")
        logger.info(f"train_f{fold:02d}")

        if args.do_train:
            model.train_1fold(fold, args.params, args.params_custom)
            model.save(is_finish=False, is_train_folds_full=is_train_folds_full)
        
    if args.do_train:
        model.save(is_finish=True, is_train_folds_full=is_train_folds_full)


if __name__ == '__main__':
    main()