import pandas as pd
import itertools
import re
import shutil
import os
import pprint as pp
import gc
import joblib
from joblib import Parallel, delayed
import argparse

import preproc_v15
import target_encorder
import util
from preproc import load, create_rank_int_features, create_folds, create_weapon_count_features, create_level_bin_features, \
    create_team_agg_features, create_category_features, create_period_features, create_team_agg_diff_features
from util import str2bool, mprof_timestamp, make_hardlink

import logging_config
from logging import getLogger
logger = getLogger("splt")


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--FOLD_NUM', type=int, default=10)
parser.add_argument('--DO_TEST', action='store_true')
parser.add_argument('--suffix', type=str, default="")
parser.add_argument('--features_path', type=str, default=None)
parser.add_argument('--in_features_path', type=str)
parser.add_argument('--recreate_basic_features', action='store_true')

parser.add_argument('--n_jobs', type=int, default=10)
parser.add_argument('--verbose_joblib', type=int, default=20)

parser.add_argument('--shuffle_folds', type=str2bool, default=True)
parser.add_argument('--stratified_folds', type=str2bool, default=False)

parser.add_argument('--fold_num_enc', type=int, default=4)  # 4
parser.add_argument('--min_samples_leaf', type=int, default=1)  # 1
parser.add_argument('--stratified_folds_enc', type=str2bool, default=False)
parser.add_argument('--do_smoothing', type=str2bool, default=True)
parser.add_argument('--handle_missing', type=str, choices=["return_nan", "prior", "value"], default="return_nan")
parser.add_argument('--handle_unknown', type=str, choices=["return_nan", "prior"], default="prior")
parser.add_argument('--do_team_agg_mode', type=str2bool, default=False)

args = parser.parse_args()

if args.DO_TEST:
    args.FOLD_NUM = 2

if args.fold_num_enc == 0:
    args.fold_num_enc = None

args.params_tarenc = {
    'fold_num_enc':args.fold_num_enc,
    'stratified_fold':args.stratified_folds_enc,
    'do_smoothing':args.do_smoothing,
    'min_samples_leaf':args.min_samples_leaf,
    'smoothing':1.0,
    'handle_missing':args.handle_missing,
    'handle_unknown':args.handle_unknown
}

args.params_freqenc = {
    'handle_missing':args.handle_missing
}

del args.fold_num_enc

timestr = util.get_time_str()

if args.features_path is None:  # 出力先を新規作成
    args.features_path = "../features/" + timestr + "-" + args.suffix
    if args.DO_TEST:
        args.features_path = "../features/test"
        util.trash(args.features_path)
    os.makedirs(args.features_path, exist_ok=False)  
else:  # 既存のdirに出力
    if args.DO_TEST:
        args.features_path = "../features/test"
    os.makedirs(args.features_path, exist_ok=True)

logging_config.init(f"{args.features_path}/log_{timestr}.log")
    
logger.info(f"features_path = {args.features_path}")

logger.info("args =\n" + pp.pformat(vars(args)))

util.dump_json(vars(args), f"{args.features_path}/args_{timestr}.json")
shutil.copytree("../src", args.features_path + "/src_" + timestr)

mprof_timestamp("basic")


def create_team_agg_features_wrp(merge):
    ma = merge[[c for c in merge.columns if re.match("A[1234]-(level|rank-int)", c)]]
    mb = merge[[c for c in merge.columns if re.match("B[1234]-(level|rank-int)", c)]]
    ma_agg = create_team_agg_features(ma, "A")
    ma234_agg = create_team_agg_features(ma, "A234")
    mb_agg = create_team_agg_features(mb, "B")

    m_diff = create_team_agg_diff_features(ma_agg, mb_agg)
    m_diff_a234b = create_team_agg_diff_features(ma234_agg, mb_agg)

    ma234_agg.columns = [c + "-tA234" for c in ma234_agg.columns]
    m_diff_a234b.columns = [c + "-tA234B" for c in m_diff_a234b.columns]

    m_all = pd.concat([ma_agg, mb_agg, ma234_agg, m_diff, m_diff_a234b], axis=1)
    return m_all


if (not args.recreate_basic_features) and os.path.exists(f"{args.in_features_path}/merge_wc.joblib"):
    if args.in_features_path != args.features_path:
        logger.info(f"Make hardlink from {args.in_features_path}")
        make_hardlink(args.in_features_path + "/merge.joblib", args.features_path)
        make_hardlink(args.in_features_path + "/merge_tm.joblib", args.features_path)
        make_hardlink(args.in_features_path + "/merge_agg.joblib", args.features_path)
        make_hardlink(args.in_features_path + "/merge_agg_v15.joblib", args.features_path)
        make_hardlink(args.in_features_path + "/merge_wc.joblib", args.features_path)
    else:
        logger.info("Pass basic features")
else:
    logger.info("Create basic features")
    merge, col_weapon_names = load(args.DO_TEST)

    merge["fold"] = create_folds(merge, args.FOLD_NUM, args.shuffle_folds, args.stratified_folds)  # Foldは事前に決めとく（TargetEncのため）

    m = create_rank_int_features(merge)
    merge[m.columns] = m

    merge_agg_v15 = preproc_v15.create_team_agg_features(merge)
    
    m = create_level_bin_features(merge, q=10)
    merge[m.columns] = m

    m = create_level_bin_features(merge, q=20)
    merge[m.columns] = m

    m = create_level_bin_features(merge, q=5)
    merge[m.columns] = m

    merge_tm = create_period_features(merge)

    merge_agg = create_team_agg_features_wrp(merge)

    merge_wc = create_weapon_count_features(merge, col_weapon_names + ["rank"])

    logger.info("Conving to category type")
    m = create_category_features(merge)
    merge[m.columns] = m

    m = create_category_features(merge_wc)
    merge_wc[m.columns] = m

    logger.info("Saving features (No TarEnc)")
    joblib.dump(merge, args.features_path + "/merge.joblib", compress=4)
    joblib.dump(merge_tm, args.features_path + "/merge_tm.joblib", compress=4)
    joblib.dump(merge_agg, args.features_path + "/merge_agg.joblib", compress=4)
    joblib.dump(merge_agg_v15, args.features_path + "/merge_agg-v15.joblib", compress=4)
    joblib.dump(merge_wc, args.features_path + "/merge_wc.joblib", compress=4)

    del merge, merge_tm, merge_agg, merge_agg_v15, merge_wc
    gc.collect()



#####################################################

def create_tarenc_features_1fold(fold, mode_target_persons, mode_target_cols :str, mode_enc :str):
    """
    Args:
        mode_target_cols:ベースの列-層別追加列
            basic-no
            basic-period
            basic-period_date
            level_bin_q10-no
            A1user-period
            など。。。
    """
    logging_config.init()

    fname = f"merge_{mode_enc}-{mode_target_persons}-{mode_target_cols}"
    if fold is not None:
        fname = fname + f"-f{fold:02d}"

    if os.path.exists(f"{args.features_path}/{fname}.joblib"):
        assert False, fname

    if os.path.exists(f"{args.in_features_path}/{fname}.joblib"):        
        if args.in_features_path != args.features_path:
            logger.info(f"Make features hardlink {fname} from {args.in_features_path}")
            make_hardlink(args.in_features_path + f"/{fname}.joblib", args.features_path)
        else:
            logger.info(f"Pass features {fname}")
        return
    
    if mode_target_persons == "all":
        do_add_cnt_list = [False, True]
        target_persons = [mode_target_persons]
    elif mode_target_persons == "NoA1":
        do_add_cnt_list = [False]
        target_persons = [mode_target_persons]
    elif mode_target_persons == "EachP":
        do_add_cnt_list = [False]
        target_persons = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]
    elif mode_target_persons == "A1":
        do_add_cnt_list = [False]
        target_persons = ["A1"]       
    elif mode_target_persons == "A234":
        do_add_cnt_list = [False]
        target_persons = [mode_target_persons]
    elif mode_target_persons == "B":
        do_add_cnt_list = [False]
        target_persons = [mode_target_persons]
    else:
        assert False, mode_target_persons

    merge, col_weapon_names = load(args.DO_TEST)
    merge["fold"] = create_folds(merge, args.FOLD_NUM, args.shuffle_folds, args.stratified_folds)

    m = create_period_features(merge)
    merge[m.columns] = m

    m = create_level_bin_features(merge, q=10)
    merge[m.columns] = m

    m = create_level_bin_features(merge, q=20)
    merge[m.columns] = m

    m = create_level_bin_features(merge, q=5)
    merge[m.columns] = m

    merge["A1-dum"] = "dum"

    if args.DO_TEST:  # 時間がかかるので間引く
        logger.info("【TEST_MODE】Remove col_weapon for tarenc.")
        col_weapon_names = ["weapon", "weapon_cat1"]

    mtc1, mtc2 = mode_target_cols.split("-")

    if mtc1 == "basic":
        col_types_per_person_all = [[c] for c in col_weapon_names + ["rank", "level"]]  # 単品
        col_types_per_person_all += [[c, "rank"]  for c in col_weapon_names]  # weapon系×rank
        col_types_per_person_all += [[c, "level"]  for c in col_weapon_names]  # weapon系×level
        col_types_per_person_all += [["rank", "level"]]  # rank×level
        col_types_per_person_all += [[c, "rank", "level"]  for c in col_weapon_names]  # weapon系×level×rank
    elif mtc1 in ["level_bin", "level_bin_q20", "level_bin_q5"]:
        if mtc1 == "level_bin":
            levels = ["level_bin_q10"]
        else:
            levels = [mtc1]
        col_types_per_person_all = [[c] for c in levels]  # 単品
        col_types_per_person_all += list(itertools.product(levels, col_weapon_names))  # weapon系×level
        col_types_per_person_all += list(itertools.product(levels, ["rank"]))  # rank×level
        col_types_per_person_all += list(itertools.product(levels, col_weapon_names, ["rank"]))  # weapon系×level×rank
    elif mtc1 == "A1user":
        levels = ["level_bin", "uid"]
        col_types_per_person_all = [[c] for c in levels]  # 単品
        col_types_per_person_all += list(itertools.product(levels, col_weapon_names))  # weapon系×level
        col_types_per_person_all += list(itertools.product(levels, ["rank"]))  # rank×level
        col_types_per_person_all += list(itertools.product(levels, col_weapon_names, ["rank"]))  # weapon系×level×rank
        do_add_cnt_list = [False]
    elif mtc1 == "no":
        col_types_per_person_all = [["dum"]]
        do_add_cnt_list = [False]
    else:
        assert False, mode_target_cols
    
    if mtc2 == "no":
        cols_per_row_all = [[], ["mode"], ["stage"], ["mode", "stage"], ['lobby-mode'], ["lobby-mode", "stage"]]
        if mtc1 == "no":
            cols_per_row_all = cols_per_row_all[1:]  # 空は除く
    elif mtc2 in ["period", "period_date", "period_2W", "period_weekday_period_hour", "period_4hour"]:
        if mtc2 == "period_weekday_period_hour":
            periods = ["period_weekday", "period_hour"]
        else:
            periods = [mtc2]
        cols_per_row_all = [[], ["mode"], ["stage"], ["mode", "stage"], ['lobby-mode'], ["lobby-mode", "stage"]]
        cols_per_row_all = [x + periods for x in cols_per_row_all]
        do_add_cnt_list = [False]
    else:
        assert False, mode_target_cols

    mts = []
    for do_add_cnt in do_add_cnt_list:
        for tp in target_persons:
            logger.info(f"Create features {mode_enc} fold={fold} cnt={do_add_cnt} tp={tp} tc={mode_target_cols}")

            if mode_enc == "tarenc":
                mt = target_encorder.create_features_1fold(
                    merge=merge, fold=fold, 
                    col_types_per_person_all=col_types_per_person_all, 
                    cols_per_row_all=cols_per_row_all,
                    do_add_cnt=do_add_cnt, target_persons=tp,
                    **args.params_tarenc)
            elif mode_enc == "freqenc":
                mt = target_encorder.create_freqenc_features(
                    merge=merge,
                    col_types_per_person_all=col_types_per_person_all, 
                    cols_per_row_all=cols_per_row_all,
                    do_add_cnt=do_add_cnt, target_persons=tp,
                    **args.params_freqenc)
            else:
                assert False, mode_enc
            
            mts.append(mt)
    
    mt_mrg = pd.concat(mts, axis=1)
    assert not mt_mrg.columns.duplicated().any()
    logger.info(f"Saving features {fname}")

    joblib.dump(mt_mrg, args.features_path + f"/{fname}.joblib", compress=4)

    return None


def create_tarenc_features(mode_target_persons, mode_target_cols):
    logger.info(f"tarenc_tp{mode_target_persons}_tc{mode_target_cols}")
    mprof_timestamp(f"tarenc_tp{mode_target_persons}_tc{mode_target_cols}")
    _ = Parallel(n_jobs=args.n_jobs, verbose=args.verbose_joblib) \
        ([delayed(create_tarenc_features_1fold)(fold, mode_target_persons, mode_target_cols, "tarenc") for fold in range(args.FOLD_NUM)])


def create_freqenc_features(mode_target_persons, mode_target_cols):
    logger.info(f"freqenc_tp{mode_target_persons}_tc{mode_target_cols}")
    mprof_timestamp(f"freqenc_tp{mode_target_persons}_tc{mode_target_cols}")
    create_tarenc_features_1fold(None, mode_target_persons, mode_target_cols, "freqenc")

######

def create_tarenc_agg_features_1fold(fold, mode_target_persons, mode_target_cols, mode_enc: str):
    logging_config.init()

    fname = f"merge_{mode_enc}_agg-{mode_target_persons}-{mode_target_cols}"
    if fold is not None:
        fname = fname + f"-f{fold:02d}"

    if os.path.exists(f"{args.features_path}/{fname}.joblib"):
        assert False, fname

    if os.path.exists(f"{args.in_features_path}/{fname}.joblib"):        
        if args.in_features_path != args.features_path:
            logger.info(f"Make features hardlink {fname} from {args.in_features_path}")
            make_hardlink(args.in_features_path + f"/{fname}.joblib", args.features_path)
        else:
            logger.info(f"Pass features {fname}")
        return

    logger.info(f"Load features for {fname}")

    if fold is None:
        f = ""
    else:
        f = f"-f{fold:02d}"

    if mode_target_persons in ["all", "EachP", "NoA1"]:
        merge = joblib.load(args.features_path + f"/merge_{mode_enc}-{mode_target_persons}-{mode_target_cols}{f}.joblib")
        ma = merge[[c for c in merge.columns if re.match("A[1234]-", c)]]
        mb = merge[[c for c in merge.columns if re.match("B[1234]-", c)]]
        if mode_target_persons == "NoA1":
            mtc_a = "A234"
        else:
            mtc_a = "A"
        itm_a = False
        itm_diff = False
        sfx_a = None
        sfx_diff = None
    elif mode_target_persons in ["A234B"]:
        ma = joblib.load(args.features_path + f"/merge_{mode_enc}-A234-{mode_target_cols}{f}.joblib")
        mb = joblib.load(args.features_path + f"/merge_{mode_enc}-B-{mode_target_cols}{f}.joblib")
        mtc_a = "A234"
        itm_a = False
        itm_diff = True
        sfx_a = None
        sfx_diff = f"-tA234B-{mode_enc}"
    elif mode_target_persons in ["A1A234B"]:
        ma1 = joblib.load(args.features_path + f"/merge_{mode_enc}-EachP-{mode_target_cols}{f}.joblib")
        ma1 = ma1[[c for c in ma1.columns if re.match("A1-", c)]]
        ma234 = joblib.load(args.features_path + f"/merge_{mode_enc}-A234-{mode_target_cols}{f}.joblib")
        ma = pd.concat([ma1, ma234], axis=1)
        mb = joblib.load(args.features_path + f"/merge_{mode_enc}-B-{mode_target_cols}{f}.joblib")
        mtc_a = "A"
        itm_a = True
        itm_diff = True
        sfx_a = f"-tA1A234-{mode_enc}"
        sfx_diff = f"-tA1A234B-{mode_enc}"
    else:
        assert False, mode_target_persons

    logger.info(f"Create features {fname}")
    m_agg_a = create_team_agg_features(ma, mtc_a, itm_a, sfx_a, do_mode=args.do_team_agg_mode)
    m_agg_b = create_team_agg_features(mb, "B", False, None, do_mode=args.do_team_agg_mode)
    m_diff = create_team_agg_diff_features(m_agg_a, m_agg_b, itm_diff, sfx_diff)
    m_agg_all = pd.concat([m_agg_a, m_agg_b, m_diff], axis=1)

    logger.info(f"Saving features {fname}")
    joblib.dump(m_agg_all, args.features_path + f"/{fname}.joblib", compress=4)


def create_tarenc_agg_features(mode_target_persons, mode_target_cols):
    logger.info(f"tarenc_agg_tp{mode_target_persons}_tc{mode_target_cols}")
    mprof_timestamp(f"tarenc_agg_tp{mode_target_persons}_tc{mode_target_cols}")
    _ = Parallel(n_jobs=args.n_jobs//2, verbose=args.verbose_joblib) \
            ([delayed(create_tarenc_agg_features_1fold)(fold, mode_target_persons, mode_target_cols, "tarenc") for fold in range(args.FOLD_NUM)])


def create_freqenc_agg_features(mode_target_persons, mode_target_cols):
    logger.info(f"freqenc_agg_tp{mode_target_persons}_tc{mode_target_cols}")
    mprof_timestamp(f"freqenc_agg_tp{mode_target_persons}_tc{mode_target_cols}")
    create_tarenc_agg_features_1fold(None, mode_target_persons, mode_target_cols, "freqenc")


#######

create_tarenc_features(mode_target_persons="A1", mode_target_cols="A1user-no")


mtc="basic-no"
create_tarenc_features(mode_target_persons="all", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="NoA1", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="EachP", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="A234", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="B", mode_target_cols=mtc)

create_tarenc_agg_features(mode_target_persons="all", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="NoA1", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="EachP", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="A234B", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="A1A234B", mode_target_cols=mtc)


mtc="level_bin-no"
create_tarenc_features(mode_target_persons="all", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="NoA1", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="EachP", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="A234", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="B", mode_target_cols=mtc)

create_tarenc_agg_features(mode_target_persons="all", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="NoA1", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="EachP", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="A234B", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="A1A234B", mode_target_cols=mtc)


mtc="basic-period_2W"
create_tarenc_features(mode_target_persons="all", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="NoA1", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="EachP", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="A234", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="B", mode_target_cols=mtc)

create_tarenc_agg_features(mode_target_persons="all", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="NoA1", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="EachP", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="A234B", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="A1A234B", mode_target_cols=mtc)


mtc="level_bin-period_2W"
create_tarenc_features(mode_target_persons="all", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="NoA1", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="EachP", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="A234", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="B", mode_target_cols=mtc)

create_tarenc_agg_features(mode_target_persons="all", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="NoA1", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="EachP", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="A234B", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="A1A234B", mode_target_cols=mtc)


mtc="basic-period_date"
create_tarenc_features(mode_target_persons="all", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="NoA1", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="EachP", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="A234", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="B", mode_target_cols=mtc)

create_tarenc_agg_features(mode_target_persons="all", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="NoA1", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="EachP", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="A234B", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="A1A234B", mode_target_cols=mtc)


mtc="level_bin-period_date"
create_tarenc_features(mode_target_persons="all", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="NoA1", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="EachP", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="A234", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="B", mode_target_cols=mtc)

create_tarenc_agg_features(mode_target_persons="all", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="NoA1", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="EachP", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="A234B", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="A1A234B", mode_target_cols=mtc)


mtc="basic-no"
create_freqenc_features(mode_target_persons="all", mode_target_cols=mtc)
create_freqenc_features(mode_target_persons="NoA1", mode_target_cols=mtc)
create_freqenc_features(mode_target_persons="EachP", mode_target_cols=mtc)
create_freqenc_features(mode_target_persons="A234", mode_target_cols=mtc)
create_freqenc_features(mode_target_persons="B", mode_target_cols=mtc)

create_freqenc_agg_features(mode_target_persons="all", mode_target_cols=mtc)
create_freqenc_agg_features(mode_target_persons="NoA1", mode_target_cols=mtc)
create_freqenc_agg_features(mode_target_persons="EachP", mode_target_cols=mtc)
create_freqenc_agg_features(mode_target_persons="A234B", mode_target_cols=mtc)
create_freqenc_agg_features(mode_target_persons="A1A234B", mode_target_cols=mtc)


mtc="level_bin-no"
create_freqenc_features(mode_target_persons="all", mode_target_cols=mtc)
create_freqenc_features(mode_target_persons="NoA1", mode_target_cols=mtc)
create_freqenc_features(mode_target_persons="EachP", mode_target_cols=mtc)
create_freqenc_features(mode_target_persons="A234", mode_target_cols=mtc)
create_freqenc_features(mode_target_persons="B", mode_target_cols=mtc)

create_freqenc_agg_features(mode_target_persons="all", mode_target_cols=mtc)
create_freqenc_agg_features(mode_target_persons="NoA1", mode_target_cols=mtc)
create_freqenc_agg_features(mode_target_persons="EachP", mode_target_cols=mtc)
create_freqenc_agg_features(mode_target_persons="A234B", mode_target_cols=mtc)
create_freqenc_agg_features(mode_target_persons="A1A234B", mode_target_cols=mtc)

###

mtc="basic-period"
create_tarenc_features(mode_target_persons="all", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="NoA1", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="EachP", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="A234", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="B", mode_target_cols=mtc)

create_tarenc_agg_features(mode_target_persons="all", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="NoA1", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="EachP", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="A234B", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="A1A234B", mode_target_cols=mtc)


mtc="level_bin-period"
create_tarenc_features(mode_target_persons="all", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="NoA1", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="EachP", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="A234", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="B", mode_target_cols=mtc)

create_tarenc_agg_features(mode_target_persons="all", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="NoA1", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="EachP", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="A234B", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="A1A234B", mode_target_cols=mtc)

###

create_tarenc_features(mode_target_persons="A1", mode_target_cols="A1user-period")
create_tarenc_features(mode_target_persons="A1", mode_target_cols="A1user-period_weekday_period_hour")

mtc="level_bin_q20-no"
create_tarenc_features(mode_target_persons="all", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="EachP", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="all", mode_target_cols=mtc)

mtc="level_bin_q20-period"
create_tarenc_features(mode_target_persons="all", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="EachP", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="all", mode_target_cols=mtc)

mtc = "basic-period_weekday_period_hour"
create_tarenc_features(mode_target_persons="all", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="EachP", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="all", mode_target_cols=mtc)

mtc = "level_bin-period_weekday_period_hour"
create_tarenc_features(mode_target_persons="all", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="EachP", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="all", mode_target_cols=mtc)

create_tarenc_features(mode_target_persons="A1", mode_target_cols="no-no")
create_tarenc_features(mode_target_persons="A1", mode_target_cols="no-period")
create_tarenc_features(mode_target_persons="A1", mode_target_cols="no-period_weekday_period_hour")

###

create_tarenc_features(mode_target_persons="A1", mode_target_cols="A1user-period_date")
create_tarenc_features(mode_target_persons="A1", mode_target_cols="A1user-period_2W")

create_tarenc_features(mode_target_persons="A1", mode_target_cols="no-period_date")
create_tarenc_features(mode_target_persons="A1", mode_target_cols="no-period_2W")

mtc="level_bin_q20-period_2W"
create_tarenc_features(mode_target_persons="all", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="EachP", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="all", mode_target_cols=mtc)

mtc="level_bin_q20-period_date"
create_tarenc_features(mode_target_persons="all", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="EachP", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="all", mode_target_cols=mtc)

###

mtc="level_bin_q20-no"
create_tarenc_features(mode_target_persons="NoA1", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="A234", mode_target_cols=mtc)
create_tarenc_features(mode_target_persons="B", mode_target_cols=mtc)

create_tarenc_agg_features(mode_target_persons="NoA1", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="EachP", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="A234B", mode_target_cols=mtc)
create_tarenc_agg_features(mode_target_persons="A1A234B", mode_target_cols=mtc)

###

create_freqenc_features(mode_target_persons="A1", mode_target_cols="A1user-no")
create_freqenc_features(mode_target_persons="A1", mode_target_cols="A1user-period")
create_freqenc_features(mode_target_persons="EachP", mode_target_cols="basic-period")
create_freqenc_features(mode_target_persons="EachP", mode_target_cols="level_bin-period")

###

create_freqenc_features(mode_target_persons="A1", mode_target_cols="no-no")
create_freqenc_features(mode_target_persons="A1", mode_target_cols="no-period")
create_freqenc_features(mode_target_persons="A1", mode_target_cols="no-period_date")
create_freqenc_features(mode_target_persons="A1", mode_target_cols="no-period_2W")
create_freqenc_features(mode_target_persons="A1", mode_target_cols="no-period_weekday_period_hour")

create_freqenc_features(mode_target_persons="EachP", mode_target_cols="basic-period_date")
create_freqenc_features(mode_target_persons="EachP", mode_target_cols="basic-period_2W")
create_freqenc_features(mode_target_persons="EachP", mode_target_cols="basic-period_weekday_period_hour")

create_freqenc_features(mode_target_persons="EachP", mode_target_cols="level_bin-period_date")
create_freqenc_features(mode_target_persons="EachP", mode_target_cols="level_bin-period_2W")
create_freqenc_features(mode_target_persons="EachP", mode_target_cols="level_bin-period_weekday_period_hour")

create_freqenc_features(mode_target_persons="A1", mode_target_cols="A1user-period_date")
create_freqenc_features(mode_target_persons="A1", mode_target_cols="A1user-period_2W")
create_freqenc_features(mode_target_persons="A1", mode_target_cols="A1user-period_weekday_period_hour")

create_freqenc_agg_features(mode_target_persons="EachP", mode_target_cols="basic-period")
create_freqenc_agg_features(mode_target_persons="EachP", mode_target_cols="basic-period_date")
create_freqenc_agg_features(mode_target_persons="EachP", mode_target_cols="basic-period_2W")
create_freqenc_agg_features(mode_target_persons="EachP", mode_target_cols="basic-period_weekday_period_hour")

create_freqenc_agg_features(mode_target_persons="EachP", mode_target_cols="level_bin-period")
create_freqenc_agg_features(mode_target_persons="EachP", mode_target_cols="level_bin-period_date")
create_freqenc_agg_features(mode_target_persons="EachP", mode_target_cols="level_bin-period_2W")
create_freqenc_agg_features(mode_target_persons="EachP", mode_target_cols="level_bin-period_weekday_period_hour")

###

create_tarenc_features(mode_target_persons="all", mode_target_cols="level_bin_q5-no")
create_tarenc_agg_features(mode_target_persons="all", mode_target_cols="level_bin_q5-no")
create_tarenc_features(mode_target_persons="EachP", mode_target_cols="level_bin_q5-no")

create_tarenc_features(mode_target_persons="all", mode_target_cols="level_bin_q5-period")
create_tarenc_agg_features(mode_target_persons="all", mode_target_cols="level_bin_q5-period")
create_tarenc_features(mode_target_persons="EachP", mode_target_cols="level_bin_q5-period")

create_tarenc_features(mode_target_persons="all", mode_target_cols="basic-period_4hour")
create_tarenc_agg_features(mode_target_persons="all", mode_target_cols="basic-period_4hour")
create_tarenc_features(mode_target_persons="EachP", mode_target_cols="basic-period_4hour")

create_tarenc_features(mode_target_persons="all", mode_target_cols="level_bin-period_4hour")
create_tarenc_agg_features(mode_target_persons="all", mode_target_cols="level_bin-period_4hour")
create_tarenc_features(mode_target_persons="EachP", mode_target_cols="level_bin-period_4hour")

create_freqenc_features(mode_target_persons="EachP", mode_target_cols="level_bin_q5-no")
create_freqenc_features(mode_target_persons="EachP", mode_target_cols="level_bin_q5-period")
create_freqenc_features(mode_target_persons="EachP", mode_target_cols="basic-period_4hour")
create_freqenc_features(mode_target_persons="EachP", mode_target_cols="level_bin-period_4hour")

###

create_tarenc_features(mode_target_persons="NoA1", mode_target_cols="level_bin_q5-no")
create_tarenc_features(mode_target_persons="A234", mode_target_cols="level_bin_q5-no")
create_tarenc_features(mode_target_persons="B", mode_target_cols="level_bin_q5-no")

create_tarenc_agg_features(mode_target_persons="NoA1", mode_target_cols="level_bin_q5-no")
create_tarenc_agg_features(mode_target_persons="A1A234B", mode_target_cols="level_bin_q5-no")

###

create_tarenc_features(mode_target_persons="NoA1", mode_target_cols="level_bin_q5-period")
create_tarenc_features(mode_target_persons="A234", mode_target_cols="level_bin_q5-period")
create_tarenc_features(mode_target_persons="B", mode_target_cols="level_bin_q5-period")

create_tarenc_agg_features(mode_target_persons="NoA1", mode_target_cols="level_bin_q5-period")
create_tarenc_agg_features(mode_target_persons="A1A234B", mode_target_cols="level_bin_q5-period")

###

create_tarenc_features(mode_target_persons="NoA1", mode_target_cols="basic-period_4hour")
create_tarenc_features(mode_target_persons="A234", mode_target_cols="basic-period_4hour")
create_tarenc_features(mode_target_persons="B", mode_target_cols="basic-period_4hour")

create_tarenc_agg_features(mode_target_persons="NoA1", mode_target_cols="basic-period_4hour")
create_tarenc_agg_features(mode_target_persons="A1A234B", mode_target_cols="basic-period_4hour")

###

create_tarenc_features(mode_target_persons="NoA1", mode_target_cols="level_bin-period_4hour")
create_tarenc_features(mode_target_persons="A234", mode_target_cols="level_bin-period_4hour")
create_tarenc_features(mode_target_persons="B", mode_target_cols="level_bin-period_4hour")

create_tarenc_agg_features(mode_target_persons="NoA1", mode_target_cols="level_bin-period_4hour")
create_tarenc_agg_features(mode_target_persons="A1A234B", mode_target_cols="level_bin-period_4hour")

###

create_freqenc_features(mode_target_persons="NoA1", mode_target_cols="level_bin_q5-no")
create_freqenc_agg_features(mode_target_persons="NoA1", mode_target_cols="level_bin_q5-no")

create_freqenc_features(mode_target_persons="NoA1", mode_target_cols="level_bin_q5-period")
create_freqenc_agg_features(mode_target_persons="NoA1", mode_target_cols="level_bin_q5-period")

create_freqenc_features(mode_target_persons="NoA1", mode_target_cols="basic-period_4hour")
create_freqenc_agg_features(mode_target_persons="NoA1", mode_target_cols="basic-period_4hour")

create_freqenc_features(mode_target_persons="NoA1", mode_target_cols="level_bin-period_4hour")
create_freqenc_agg_features(mode_target_persons="NoA1", mode_target_cols="level_bin-period_4hour")

###

create_tarenc_agg_features(mode_target_persons="EachP", mode_target_cols="level_bin_q5-no")
create_tarenc_agg_features(mode_target_persons="A234B", mode_target_cols="level_bin_q5-no")

create_tarenc_agg_features(mode_target_persons="EachP", mode_target_cols="level_bin_q5-period")
create_tarenc_agg_features(mode_target_persons="A234B", mode_target_cols="level_bin_q5-period")

create_tarenc_agg_features(mode_target_persons="EachP", mode_target_cols="basic-period_4hour")
create_tarenc_agg_features(mode_target_persons="A234B", mode_target_cols="basic-period_4hour")

create_tarenc_agg_features(mode_target_persons="EachP", mode_target_cols="level_bin-period_4hour")
create_tarenc_agg_features(mode_target_persons="A234B", mode_target_cols="level_bin-period_4hour")

###

create_freqenc_features(mode_target_persons="all", mode_target_cols="level_bin_q5-no")
create_freqenc_features(mode_target_persons="A234", mode_target_cols="level_bin_q5-no")
create_freqenc_features(mode_target_persons="B", mode_target_cols="level_bin_q5-no")

create_freqenc_agg_features(mode_target_persons="all", mode_target_cols="level_bin_q5-no")
create_freqenc_agg_features(mode_target_persons="EachP", mode_target_cols="level_bin_q5-no")
create_freqenc_agg_features(mode_target_persons="A234B", mode_target_cols="level_bin_q5-no")
create_freqenc_agg_features(mode_target_persons="A1A234B", mode_target_cols="level_bin_q5-no")


create_freqenc_features(mode_target_persons="all", mode_target_cols="level_bin_q5-period")
create_freqenc_features(mode_target_persons="A234", mode_target_cols="level_bin_q5-period")
create_freqenc_features(mode_target_persons="B", mode_target_cols="level_bin_q5-period")

create_freqenc_agg_features(mode_target_persons="all", mode_target_cols="level_bin_q5-period")
create_freqenc_agg_features(mode_target_persons="EachP", mode_target_cols="level_bin_q5-period")
create_freqenc_agg_features(mode_target_persons="A234B", mode_target_cols="level_bin_q5-period")
create_freqenc_agg_features(mode_target_persons="A1A234B", mode_target_cols="level_bin_q5-period")


create_freqenc_features(mode_target_persons="all", mode_target_cols="basic-period_4hour")
create_freqenc_features(mode_target_persons="A234", mode_target_cols="basic-period_4hour")
create_freqenc_features(mode_target_persons="B", mode_target_cols="basic-period_4hour")

create_freqenc_agg_features(mode_target_persons="all", mode_target_cols="basic-period_4hour")
create_freqenc_agg_features(mode_target_persons="EachP", mode_target_cols="basic-period_4hour")
create_freqenc_agg_features(mode_target_persons="A234B", mode_target_cols="basic-period_4hour")
create_freqenc_agg_features(mode_target_persons="A1A234B", mode_target_cols="basic-period_4hour")


create_freqenc_features(mode_target_persons="all", mode_target_cols="level_bin-period_4hour")
create_freqenc_features(mode_target_persons="A234", mode_target_cols="level_bin-period_4hour")
create_freqenc_features(mode_target_persons="B", mode_target_cols="level_bin-period_4hour")

create_freqenc_agg_features(mode_target_persons="all", mode_target_cols="level_bin-period_4hour")
create_freqenc_agg_features(mode_target_persons="EachP", mode_target_cols="level_bin-period_4hour")
create_freqenc_agg_features(mode_target_persons="A234B", mode_target_cols="level_bin-period_4hour")
create_freqenc_agg_features(mode_target_persons="A1A234B", mode_target_cols="level_bin-period_4hour")

#######################################

logger.info("Finish all")
