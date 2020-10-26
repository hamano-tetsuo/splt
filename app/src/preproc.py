import re
import numpy as np
import pandas as pd
import itertools
from collections import OrderedDict
from tqdm.auto import tqdm
import datetime

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer

from logging import getLogger
logger = getLogger("splt")


def load(DO_TEST = False):
    """raw csvとweaponなど外部データを読み込んでjoin"""

    train = pd.read_csv("../data/train_data.csv")
    test = pd.read_csv('../data/test_data.csv')

    if DO_TEST:
        logger.info("【TEST_MODE】Remove train/test data.")
        train = train.iloc[:1000,:]
        test = test.iloc[:300,:]

    train["split_type"] = "train"
    test["split_type"] = "test"

    merge = pd.concat([train, test]).reset_index(drop=True).drop(["game-ver", "lobby"], axis=1)   

    merge["period"] = pd.to_datetime(merge["period"])

    # 外部データの結合
    weapon = pd.read_csv("../data/weapon_merge.csv")
    weapon = weapon[["key", "category1", "category2", "mainweapon", "subweapon", "special", "reskin", "main_power_up"]]
    weapon.columns = ["key", "cat1", "cat2", "main", "sub", "special", "reskin", "powerup"]
    
    # 区切り文字が無いことを確認
    assert weapon.applymap(lambda x:"-" in x).any().any() == False
    assert merge[["mode", "stage"]].applymap(lambda x:"-" in x).any().any() == False

    m2 = merge.copy()

    for team, num in itertools.product(["A", "B"], [1, 2, 3, 4]):
        m2 = pd.merge(left=m2, right=weapon, left_on=f"{team}{num}-weapon", right_on="key", how="left")
        assert m2.shape[0] == merge.shape[0]
        m2 = m2.drop("key", axis=1).rename(columns={x:f"{team}{num}-weapon_{x}" for x in weapon.columns if x!= "key"})

    col_weapon_names = ["weapon"] + ["weapon_" + x for x in weapon.columns[1:]]

    # 外部stageデータの結合
    stage = pd.read_csv("../data/stage.csv")
    area = pd.merge(left=merge["stage"], right=stage, left_on="stage", right_on="key", how="left")["area"]
    m2["stage_area"] = area

    # A1 user推定関連
    m = pd.read_csv("../data/merge_A1-level_bin.csv")
    m2["A1-level_bin"] = m["A1-level_bin"]

    m = pd.read_csv("../data/merge_A1-uid.csv")
    m2["A1-uid"] = m["A1-uid"]

    return m2, col_weapon_names


def create_folds(merge: pd.DataFrame, fold_num: int, shuffle: bool, stratified: bool, seed=42):
    merge_f = merge[["split_type", "y"]].copy()

    if stratified:
        kf = StratifiedKFold(n_splits=fold_num, random_state=seed, shuffle=shuffle)
    else:
        kf = KFold(n_splits=fold_num, random_state=seed, shuffle=shuffle)

    merge_f["fold"] = pd.NA

    train = merge_f.loc[merge_f["split_type"] == "train"]
    train_x = train.drop("y", axis=1)
    train_y = train["y"]
    for i, (_, vdx) in enumerate(kf.split(train_x, train_y)):
        assert merge_f.loc[vdx, "fold"].isnull().all()
        merge_f.loc[vdx, "fold"] = i

    merge_f = merge_f[["fold"]]
    return merge_f


def create_period_features(merge):
    merge = merge[["period"]].copy()

    merge_tm = pd.DataFrame({}, index = merge.index)
    merge_tm["period_month"] = merge["period"].dt.month.replace({10:0, 11:1, 12:2, 1:3})
    merge_tm["period_day"] = merge["period"].dt.day
    merge_tm["period_weekday"] = merge["period"].dt.weekday
    merge_tm["period_hour"] = merge["period"].dt.hour

    # 以下は通しで振る
    merge_tm["period_date"] = (merge["period"].dt.date - merge["period"].dt.date.min()).dt.days + 1

    min_tm = merge["period"].min()
    merge_tm["period_2W"] = pd.NA
    interval = datetime.timedelta(weeks=2)
    for i in range(100):
        cond = ((min_tm + i*interval) <= merge["period"]) & (merge["period"] < (min_tm + (i+1)*interval))
        merge_tm.loc[cond, "period_2W"] = i
    merge_tm["period_2W"] = merge_tm["period_2W"].replace({6:5})  # 半端なのでマージする
    assert not merge_tm["period_2W"].isnull().any()

    min_tm = merge["period"].min()
    merge_tm["period_4hour"] = pd.NA
    interval = datetime.timedelta(hours=4)
    for i in range(1000):
        cond = ((min_tm + i*interval) <= merge["period"]) & (merge["period"] < (min_tm + (i+1)*interval))
        merge_tm.loc[cond, "period_4hour"] = i
    assert not merge_tm["period_4hour"].isnull().any()
    merge_tm["period_4hour"] = merge_tm["period_4hour"].astype(int)

    # periodをint変換(hour単位なら解像度落ちない)
    td = (merge["period"] - merge["period"].min()).dt
    sec = td.days * 24 * 60 * 60 + td.seconds
    pd.testing.assert_series_equal(sec, td.total_seconds().astype(int))
    assert (sec.mod(60*60) == 0).all()
    merge_tm["period_total_hours"] = (sec//(60*60))

    return merge_tm


def create_rank_int_features(merge):
    merge_ri = pd.DataFrame({}, index = merge.index)

    rank_order = ['x', 's+', 's', 'a+', 'a', 'a-', 'b+', 'b', 'b-', 'c+', 'c', 'c-']
    rank_map = {x:i for i, x in enumerate(rank_order[::-1])}
    rank_map[np.nan] = np.nan

    cols_rank = [x for x in merge.columns if re.search("^[AB][1234]-rank$", x)]
    unique_rank = set(merge[cols_rank].stack().unique()) | {np.nan}
    assert set(rank_map.keys()) == unique_rank

    cols_rank_int = [x + "-int" for x in cols_rank]
    merge_ri[cols_rank_int] = merge[cols_rank].replace(rank_map)

    return merge_ri


def create_level_bin_features(_merge, q):
    cols = [x for x in _merge.columns if re.fullmatch("[AB][1234]-level", x)]
    merge = _merge[cols].copy()
    merge["index"] = merge.index
    ms = merge.set_index(["index"]).stack().reset_index()
    ms.columns = ["index", "col", "val"]
    ms["val"] = pd.qcut(ms["val"], q)
    ms["val"] = ms["val"].cat.codes  # 小さい順に0,1,2,...
    ms_bin = ms.pivot(index="index", columns="col", values="val")
    ms_bin.columns = [c + f"_bin_q{q}" for c in ms_bin.columns]
    ms_bin = ms_bin.sort_index()
    return ms_bin


class MyDataFrame:
    """始めにデータ確保して列を順に追加していくのに使う。"""
    def __init__(self, index, cols_num):
        self.dat = np.ones((len(index), cols_num)) * np.nan
        self.col_names = []
        self.index = index

    def add(self, col_name, col_data):
        assert (len(col_data.shape) == 1) or (col_data.shape[1] == 1)  # 1列であること
        ci = len(self.col_names)
        self.dat[:, ci] = col_data
        self.col_names.append(col_name)

    def get(self, col_name):
        ci = self.col_names.index(col_name)
        return self.dat[:, ci]

    def fix(self):
        # 始めに多めに確保しておいて後で削る、ならここ緩和する
        assert self.dat.shape[1] == len(self.col_names), f"dat.shape={self.dat.shape} len(col_names)={len(self.col_names)}"
        dat = pd.DataFrame(self.dat, index=self.index, columns=self.col_names)
        return dat


def create_team_agg_features(merge, mode_target_cols, ignore_tarenc_mode=False, suffix=None, do_mode=True):
    if mode_target_cols == "A":
        base_cols = [c[3:] for c in merge.columns if re.match("A2-", c)]  # A1user系除くため
    elif mode_target_cols == "B":
        base_cols = [c[3:] for c in merge.columns if re.match("B1-", c)]
    elif mode_target_cols == "A234":
        base_cols = [c[3:] for c in merge.columns if re.match("A2-", c)]
    else:
        assert False

    logger.debug(f"base_cols[:5] = {base_cols[:5]}")

    col_agg_names = ["max", "min", "mean", "median", "modes"]
    if not do_mode:
        col_agg_names = col_agg_names[:-1]   

    # 先にデータ確保
    cols_num = len(base_cols) * len(col_agg_names)
    merge_agg = MyDataFrame(merge.index, cols_num)

    with tqdm(base_cols, desc="TeamAgg") as pbar:
        for base_col in pbar:
            if mode_target_cols == "A":
                team = "A"
                pids = "1234"
            elif mode_target_cols == "B":
                team = "B"
                pids = "1234"
            elif mode_target_cols == "A234":
                team = "A"
                pids = "234"
            else:
                assert False      
            
            if ignore_tarenc_mode:
                fe = re.search("(.+)-t(.+)-(tar|freq)enc", base_col).group(1)
                cs = [c for c in merge.columns if re.fullmatch(f"{team}[{pids}]-{fe}-t(.+)-(tar|freq)enc", c)]
            else:
                fe = base_col
                cs = [c for c in merge.columns if re.fullmatch(f"{team}[{pids}]-{base_col}", c)]
                if suffix is None:
                    suffix = ""

            assert suffix is not None
            
            logger.debug(cs)

            if "-addcnt-" in cs[0]:
                assert len(cs) <= len(pids), cs
            else:
                assert len(cs) == len(pids), cs

            merge_agg.add(f"{team}-{fe}{suffix}-max", merge[cs].max(axis=1))
            merge_agg.add(f"{team}-{fe}{suffix}-min", merge[cs].min(axis=1))
            merge_agg.add(f"{team}-{fe}{suffix}-mean", merge[cs].mean(axis=1))
            merge_agg.add(f"{team}-{fe}{suffix}-median", merge[cs].median(axis=1))
            if do_mode:
                merge_agg.add(f"{team}-{fe}{suffix}-modes", merge[cs].mode(axis=1).mean(axis=1))

    merge_agg = merge_agg.fix()
    return merge_agg


def create_team_agg_diff_features(merge_agg_a, merge_agg_b, ignore_tarenc_mode=False, suffix=None):
    merge_agg_diff = MyDataFrame(merge_agg_a.index, merge_agg_a.shape[1])

    for ca in tqdm(merge_agg_a.columns, desc="TeamAggDiff"):
        logger.debug(f"ca={ca}")
        if ignore_tarenc_mode:
            mt = re.search("^A-(.+)-t(.+)-(tar|freq)enc-(mean|max|min|median|modes)$", ca)
            logger.debug(mt.groups())
            fe = mt.group(1)
            ag = mt.group(4)
            cs = [c for c in merge_agg_b.columns if re.fullmatch(f"B-{fe}-t(.+)-(tar|freq)enc-{ag}", c)]
            assert suffix is not None
            col_diff = f"Diff-{fe}{suffix}-{ag}"
            assert len(cs) == 1, f"ca={ca} fe={fe} ag={ag} cs={cs}"
        else:
            fe = re.sub("^A-", "", ca)
            cs = [c for c in merge_agg_b.columns if re.fullmatch(f"B-{fe}", c)]
            col_diff = f"Diff-{fe}"
            assert len(cs) == 1, f"ca={ca} fe={fe} cs={cs}"

        cb = cs[0]

        logger.debug(f"cb={cb} cd={col_diff}")
        merge_agg_diff.add(col_diff, merge_agg_a[ca] - merge_agg_b[cb])

    merge_agg_diff = merge_agg_diff.fix()
    return merge_agg_diff   


def create_weapon_count_features(merge, col_weapon_names):
    """Teamごとにweaponをcount。つまり1,2,3,4の区別はなくす。"""

    cols_weapon_all = []
    cols_weapon_difftype_all = []

    merge_wc = pd.DataFrame({}, index = merge.index)

    logger.info(f"col_weapon_names = {col_weapon_names}")
    with tqdm(col_weapon_names, desc="WeaponCnt") as pbar:
        for w in pbar:
            pbar.set_postfix(OrderedDict(name={f"{w}"}))
            
            # チーム内　ブキcount
            cols_weapon = [x for x in merge.columns if re.search(f"^[AB][1234]-{w}$", x)]
            cols_weapon_all += cols_weapon

            cvec = CountVectorizer(analyzer=lambda x:x, lowercase=False)
            cvec.fit([sorted(merge[cols_weapon].stack().dropna().unique())])

            for team in ["A", "B"]:
                d = cvec.transform(merge[[x for x in merge.columns if re.search(f"^{team}[1234]-{w}$", x)]].values).toarray()
                for i, k in enumerate(cvec.vocabulary_.keys()):
                    merge_wc[f"{team}-{w}-cnt-{k}"] = d[:, i]
            
            # チーム間　ブキcount 差分
            for k in cvec.vocabulary_.keys():
                merge_wc[f"Diff-{w}-cnt-{k}"] = merge_wc[f"A-{w}-cnt-{k}"] - merge_wc[f"B-{w}-cnt-{k}"]
            
            # Aにしかない、Bにしかない、両方にある、両方に無い、の4パターン
            for k in cvec.vocabulary_.keys():
                merge_wc[f"DiffType-{w}-cnt-{k}"] = pd.NA  # np.nanより全然早い
                cond1 = merge_wc[f"A-{w}-cnt-{k}"] > 0
                cond2 = merge_wc[f"B-{w}-cnt-{k}"] > 0
                merge_wc.loc[cond1 & cond2,  f"DiffType-{w}-cnt-{k}"] = "AB"
                merge_wc.loc[cond1 & ~cond2,  f"DiffType-{w}-cnt-{k}"] = "A"
                merge_wc.loc[~cond1 & cond2,  f"DiffType-{w}-cnt-{k}"] = "B"
                merge_wc.loc[~cond1 & ~cond2,  f"DiffType-{w}-cnt-{k}"] = "N"
            
            cols_weapon_difftype = [x for x in merge_wc.columns if re.search(f"^DiffType-{w}-cnt-", x)]
            cols_weapon_difftype_all += cols_weapon_difftype

    return merge_wc


def create_category_features(merge):
    """pandas category型に変換。dtypes見て判定でもいいかも。"""
    cols = []
    cols += [x for x in merge.columns if x in ['lobby-mode', 'mode', 'stage']]
    cols += [x for x in merge.columns if re.match("[AB][1234]-weapon", x)]
    cols += [x for x in merge.columns if re.match("DiffType-weapon", x)]
    cols += [x for x in merge.columns if re.match("DiffType-rank", x)]
    merge_cat = merge[cols].astype("category").copy()
    return merge_cat