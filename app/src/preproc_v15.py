import re
import numpy as np
import pandas as pd
import itertools
from tqdm.auto import tqdm

from logging import getLogger
logger = getLogger("splt")


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
        assert self.dat.shape[1] == len(self.col_names)  # 始めに多めに確保しておいて後で削る、ならここ緩和する
        dat = pd.DataFrame(self.dat, index=self.index, columns=self.col_names)
        return dat


def create_team_agg_features(merge, do_mode=True):
    """Teamごとに数値特徴量を集約。"""
    cols = [x[3:] for x in merge.columns if re.match("B1-", x) and merge.dtypes[x] in [int, float]]

    col_agg_names = ["max", "min", "mean", "median", "modes"]
    if not do_mode:
        col_agg_names = col_agg_names[:-1]

    # 先にデータ確保
    cols_num = (2 * len(cols) * len(col_agg_names)) + (len(cols) * len(col_agg_names))
    merge_agg = MyDataFrame(merge.index, cols_num)
    
    with tqdm(itertools.product(["A", "B"], cols), total=2*len(cols), desc="TeamAgg") as pbar:
        for team, fe in pbar:
            cs = [x for x in merge.columns if re.search(rf"{team}\d-{fe}", x)]
            merge_agg.add(f"{team}-{fe}-max", merge[cs].max(axis=1))
            merge_agg.add(f"{team}-{fe}-min", merge[cs].min(axis=1))
            merge_agg.add(f"{team}-{fe}-mean", merge[cs].mean(axis=1))
            merge_agg.add(f"{team}-{fe}-median", merge[cs].median(axis=1))
            if do_mode:
                merge_agg.add(f"{team}-{fe}-modes", merge[cs].mode(axis=1).mean(axis=1))

    # チーム間の差分
    for fe, ag in tqdm(itertools.product(cols, col_agg_names), total=len(cols)*len(col_agg_names), desc="TeamAggDiff"):
        merge_agg.add(f"Diff-{fe}-{ag}", merge_agg.get(f"A-{fe}-{ag}") - merge_agg.get(f"B-{fe}-{ag}"))

    merge_agg = merge_agg.fix()
    return merge_agg
