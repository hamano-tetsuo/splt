"""
TargetEncoding, FrequencyEncoding
"""
import re
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm.auto import tqdm
from pandas.api.types import is_datetime64_any_dtype as is_datetime

from logging import getLogger
logger = getLogger("splt")


def stack(dat):
    dat_st = dat.set_index(["index", "y"]).stack().reset_index()
    dat_st.columns = ["index", "y"] + ["col", "val"]
    dat_st["team"] = dat_st.col.str[:1]
    dat_st.loc[dat_st["team"] == "B", "y"] = 1 - dat_st.loc[dat_st["team"] == "B", "y"]
    return dat_st


def add_cnt(dat_st):
    temp = dat_st.groupby(["index", "team", "y", "val"]).count()
    temp = temp.reset_index()
    temp["val2"] = temp["val"] + "--" + temp["col"].astype(str)
    temp["id_in_team"] = temp.groupby(["index", "team"]).cumcount() + 1
    temp["team-id"] = temp["team"] + temp["id_in_team"].astype(str)
    dat_st_2 = temp[["index", "team", "y", "val2"]].rename(columns={"val2":"val"})
    return dat_st_2


def unstack(vl, index):
    vl["id_in_team"] = vl.groupby(["index", "team"]).cumcount() + 1
    vl["team-id"] = vl["team"] + vl["id_in_team"].astype(str)
    vl = vl.pivot(index='index', columns="team-id", values="val")
    vl = vl.reindex(index)
    return vl


def concat_cols(tr, col_types_per_person, target_persons):
    """人ごとの特徴量の結合 rankとweaponとか"""
    cols_per_person_new = []
    if target_persons == "all":
        team_pid_pairs = itertools.product(["A", "B"], range(1, 5))
    elif target_persons == "NoA1":
        team_pid_pairs = list(itertools.product(["A"], [2,3,4])) + list(itertools.product(["B"], [1,2,3,4]))
    elif target_persons in ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]:
        team_pid_pairs = [[target_persons[0], target_persons[1]]]
    elif target_persons == "A234":
        team_pid_pairs = list(itertools.product(["A"], [2,3,4]))
    elif target_persons == "B":
        team_pid_pairs = list(itertools.product(["B"], [1,2,3,4]))
    else:
        assert False, target_persons

    for team, i in team_pid_pairs:
        col_mrg = f'{team}{i}-{"_".join(col_types_per_person)}'
        tr[col_mrg] = ''
        d = tr[[f"{team}{i}-{c}" for c in col_types_per_person]].fillna("NA")  # 欠損も1つの水準として扱う
        tr[col_mrg] = tr[col_mrg].str.cat(d, "--")
        cols_per_person_new.append(col_mrg)
    tr = tr[["index", "y"] + cols_per_person_new]
    return tr


def fit_transform_weapon(train: pd.DataFrame, valid: pd.DataFrame, 
                        col_types_per_person: list, cols_per_row: list, 
                        target_persons: str,
                        do_add_cnt: bool, do_smoothing: bool, min_samples_leaf: float, smoothing: float, handle_missing: str, handle_unknown: str,
                        mode_enc: str):
    """
    Args:
        col_types_per_person: A1-XXXなど、人ごとにある特徴量のXXの部分 ex)['weapon_cat1', 'rank']
        cols_per_row: 行ごとにある特徴量　ex)['mode']
        do_add_cnt: Trueの場合、特徴量のTeam内の数でencodeする
        target_persons: 集計範囲の指定 all=A1234 B1234 全部mergeしてencode、など
        handle_unknown: 'return_nan' 'prior'
        handle_missing: 'return_nan' 'value' 'prior'
        mode_enc: 'tarenc' 'freqenc'

    References:
        https://contrib.scikit-learn.org/category_encoders/targetencoder.html
    """
    assert handle_missing in ["return_nan", "prior", "value"]
    assert handle_unknown in ["return_nan", "prior"]
    
    UNKNOWN = "[UNK]"
    NULL = "[NULL]"
    
    cols_per_person = []
    for col in col_types_per_person:
        if target_persons == "all":
            pat = "[AB][1234]"
        elif target_persons == "NoA1":
            pat = "((A[234])|(B[1234]))"
        elif target_persons in ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]:
            pat = target_persons
        elif target_persons == "A234":
            pat = "A[234]"
        elif target_persons == "B":
            pat = "B[1234]"
        else:
            assert False, target_persons
        cols_per_person += [x for x in train.columns if re.search(f"^{pat}-{col}$", x)]

    tr = train[["y"] + cols_per_person].copy()
    vl = valid[cols_per_person].copy()
    vl["y"] = 0

    tr2 = train[cols_per_row].copy()
    vl2 = valid[cols_per_row].copy()

    tr["index"] = tr.index
    vl["index"] = vl.index

    for c in cols_per_person:
        if (tr[c].dtype in [float, int]) or is_datetime(tr[c]):
            tr[c] = tr[c].apply(lambda x: x if pd.isnull(x) else str(x))
            vl[c] = vl[c].apply(lambda x: x if pd.isnull(x) else str(x))

    for c in cols_per_row:
        if (tr2[c].dtype in [float, int]) or is_datetime(tr2[c]):
            tr2[c] = tr2[c].apply(lambda x: x if pd.isnull(x) else str(x))
            vl2[c] = vl2[c].apply(lambda x: x if pd.isnull(x) else str(x))
    
    if len(col_types_per_person) > 1:
        tr = concat_cols(tr, col_types_per_person, target_persons)
        vl = concat_cols(vl, col_types_per_person, target_persons)
        cols_per_person = tr.columns[2:]
    
    # 行ごとの特徴と人ごとの特徴を結合
    if cols_per_row:
        for c in cols_per_person:
            tr[c] = tr[c].str.cat(tr2[cols_per_row], "--")  # mode, stageにnull無し
            vl[c] = vl[c].str.cat(vl2[cols_per_row], "--")

    if handle_missing == "value":
        for c in cols_per_person:
            tr[c] = tr[c].fillna("NA")
            vl[c] = vl[c].fillna("NA")

    train_st = stack(tr)

    if do_add_cnt:
        train_st = add_cnt(train_st)
    
    train_st = train_st.drop_duplicates(subset=["index", "team", "val"])
    
    stats_mean = train_st.groupby("val")["y"].mean()
    stats_cnt = train_st.groupby("val")["index"].count()
              
    prior = 0.5  # TODO:
    
    if mode_enc == "tarenc":
        if do_smoothing:
            smoove = 1 / (1 + np.exp(-(stats_cnt - min_samples_leaf) / smoothing))
            smoothing = prior * (1 - smoove) + stats_mean * smoove
        else:
            smoothing = dict()
            for k, v in stats_cnt.items():
                if v >= min_samples_leaf:  # 閾値以下の頻度はunknown相当の扱いにする
                    smoothing[k] = stats_mean[k]
            smoothing = pd.Series(smoothing)
    elif mode_enc == "freqenc":
        smoothing = stats_cnt
    else:
        assert False, mode_enc

    if handle_unknown == 'return_nan':
        smoothing[UNKNOWN] = np.nan
    elif handle_unknown == 'prior':
        smoothing[UNKNOWN] = prior

    if handle_missing == 'return_nan':
        smoothing[NULL] = np.nan
    elif handle_missing == 'prior':
        smoothing[NULL] = prior

    mapping = smoothing.to_dict()

    if do_add_cnt:
        vl = unstack(add_cnt(stack(vl)), valid.index)
        cs = "_".join(col_types_per_person)
        vl.columns = [x + f"-{cs}" for x in vl.columns]
    else:
        vl = vl.drop(["index", "y"], axis=1)

    if cols_per_row:
        sfx_pr = "_".join(cols_per_row)
    else:
        sfx_pr = "no"

    if do_add_cnt:
        sfx_ac = "addcnt"
    else:
        sfx_ac = "no"

    if target_persons == "all":
        sfx_tp = "tAll"
    elif target_persons == "NoA1":
        sfx_tp = "tNoA1"
    elif target_persons in ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]:
        sfx_tp = "tP"
    elif target_persons == "A234":
        sfx_tp = "tA234"
    elif target_persons == "B":
        sfx_tp = "tB"
    else:
        assert False

    vl.columns = [x + f"-{sfx_pr}-{sfx_ac}-{sfx_tp}-{mode_enc}" for x in vl.columns]

    for c in vl.columns:
        vl[c] = vl[c].fillna(NULL)
        vl[c] = vl[c].apply(lambda x:x if x in mapping else UNKNOWN)
        vl[c] = vl[c].apply(lambda x:mapping[x])
    
    assert (vl.index == valid.index).all()
    
    return vl, mapping, stats_mean, stats_cnt


def create_freqenc_features(merge: pd.DataFrame, col_types_per_person_all: list, cols_per_row_all: list, 
                            target_persons: str, do_add_cnt: bool, handle_missing: str):
    merge_freqenc_all = []

    def _fit_transform_weapon(tr, vl, col_types_per_person, cols_per_row):
        handle_unknown = "return_nan"  # 発生しないはず
        do_smoothing, min_samples_leaf, smoothing = None, None, None  # 使わない
        return fit_transform_weapon(tr, vl, col_types_per_person, cols_per_row, target_persons,
                        do_add_cnt, do_smoothing, min_samples_leaf, smoothing, handle_missing, handle_unknown, "freqenc")
    
    for csp, csr in tqdm(itertools.product(col_types_per_person_all, cols_per_row_all), \
                        total=len(col_types_per_person_all)*len(cols_per_row_all), desc=f"FreqEnc col", position=0):
        mg_enced, *_ = _fit_transform_weapon(merge, merge, csp, csr)
        merge_freqenc_all.append(mg_enced)
    
    merge_freqenc_all = pd.concat(merge_freqenc_all, axis=1)

    assert (merge_freqenc_all.index == merge.index).all()
    return merge_freqenc_all


def create_features_1fold(merge: pd.DataFrame, fold: int, 
                            col_types_per_person_all: list, cols_per_row_all: list, 
                            target_persons: str,
                            do_add_cnt: bool, do_smoothing: bool,
                            fold_num_enc: int, stratified_fold: bool,
                            min_samples_leaf: float, smoothing: float, handle_missing: str, handle_unknown: str):
    """
    Args:
        col_types_per_person_all: ex) [["weapon_cat1"], ["weapon_cat1", "rank"]]
        cols_per_row_all: ex) [[], ["mode"]]
        fold_num_enc: Noneだとgreedy、1以上だとHoldOutでTargetEncodeする。
        stratified_fold: HoldOutの時にStratifiedKFoldでやるか否か
    """

    train = merge.loc[merge["split_type"] == "train"]
    test = merge.loc[merge["split_type"] == "test"]

    # train dataのencode #####
    train_tarenc_all = []

    tr = train.loc[train["fold"] != fold, :]
    vl = train.loc[train["fold"] == fold, :]

    def _fit_transform_weapon(tr, vl, col_types_per_person, cols_per_row):
        return fit_transform_weapon(tr, vl, col_types_per_person, cols_per_row, target_persons,
                        do_add_cnt, do_smoothing, min_samples_leaf, smoothing, handle_missing, handle_unknown, "tarenc")
    
    for csp, csr in tqdm(itertools.product(col_types_per_person_all, cols_per_row_all), \
                        total=len(col_types_per_person_all)*len(cols_per_row_all), desc=f"TarEnc f{fold:02d} col", position=0):
        train_tarenc = []
        
        # train dataのencode
        if fold_num_enc:  # hold out
            if stratified_fold:
                kf_enc = StratifiedKFold(n_splits=fold_num_enc, random_state=100+fold, shuffle=True)
            else:
                kf_enc = KFold(n_splits=fold_num_enc, random_state=100+fold, shuffle=True)
            
            for idx_tr_enc, idx_vl_enc in kf_enc.split(tr, tr["y"]):
                tr_enc = tr.iloc[idx_tr_enc]
                vl_enc = tr.iloc[idx_vl_enc]
                vl_enced, *_ = _fit_transform_weapon(tr_enc, vl_enc, csp, csr)
                train_tarenc.append(vl_enced)
        else:  # greedy
            tr_enced, *_ = _fit_transform_weapon(tr, tr, csp, csr)
            train_tarenc.append(tr_enced)
        
        # valid dataのencode
        vl_enced, *_ = _fit_transform_weapon(tr, vl, csp, csr)
        train_tarenc.append(vl_enced)
        
        train_tarenc = pd.concat(train_tarenc).sort_index()
        assert (train_tarenc.index == train.index).all()
        train_tarenc_all.append(train_tarenc)
    
    train_tarenc_all = pd.concat(train_tarenc_all, axis=1)
            
    # test dataのencode #####
    test_tarenc_fold0 = []
    for csp, csr in tqdm(itertools.product(col_types_per_person_all, cols_per_row_all), \
                        total=len(col_types_per_person_all)*len(cols_per_row_all), desc="TarEnc test"):
        test_enced, *_ = _fit_transform_weapon(train, test, csp, csr)
        test_tarenc_fold0.append(test_enced)
            
    test_tarenc_fold0 = pd.concat(test_tarenc_fold0, axis=1, sort=False)

    test_tarenc_all = test_tarenc_fold0

    # valid #####
    assert (train_tarenc_all.index == train.index).all()
    assert (test_tarenc_all.index == test.index).all()
    assert all(train_tarenc_all.columns == test_tarenc_all.columns)

    merge_tarenc = pd.concat([train_tarenc_all, test_tarenc_all])
    assert (merge_tarenc.index == merge.index).all()
    return merge_tarenc
