import numpy as np
import pandas as pd
import re
import copy
from collections import defaultdict
import joblib
import gc
from pandas.api.types import is_numeric_dtype, is_categorical_dtype, is_string_dtype
import time
from tqdm.auto import tqdm

from sklearn.metrics import accuracy_score, log_loss
import lightgbm as lgb
from lightgbm.callback import _format_eval_result
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import scipy.stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.nn import BCEWithLogitsLoss

from IPython.display import display

import util
from util import mprof_timestamp

import logging
from logging import getLogger
logger = getLogger("splt")


def log_evaluation(logger, period=1, show_stdv=True, level=logging.INFO):
    def _callback(env):
        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
            result = '\t'.join([_format_eval_result(x, show_stdv) for x in env.evaluation_result_list])
            logger.log(level, '[{}]\t{}'.format(env.iteration+1, result))
    _callback.order = 10
    return _callback


def accuracy_lgb(preds, data):
    acc = accuracy_score(data.get_label(), np.round(preds))

    # eval_name, eval_result, is_higher_better
    return 'accuracy', acc, True


def logloss_lgb(preds, data):
    loss = log_loss(data.get_label(), preds)

    # eval_name, eval_result, is_higher_better
    return 'logloss', loss, False


def build_feval(params_custom):
    # early stoppingで使う指標は先頭にする
    if params_custom["early_stopping_metrics"] == "accuracy":
        feval = lambda preds, data:[accuracy_lgb(preds, data), logloss_lgb(preds, data)]
        #feval_names = ["accuracy", "logloss"]
    elif params_custom["early_stopping_metrics"] == "logloss":
        feval = lambda preds, data:[logloss_lgb(preds, data), accuracy_lgb(preds, data)]
        #feval_names = ["logloss", "accuracy"]
    else:
        assert False
    return feval


def mets2str(acc_t, acc_v, loss_t, loss_v):
    return f"valid's acc: {acc_v:.6g}\tvalid's logloss: {loss_v:.6g}\ttrain's acc: {acc_t:.6g}\ttrain's logloss: {loss_t:.6g}"


def show_mets(fold, acc_t, acc_v, loss_t, loss_v, best_iter):
    s = f"Best iteration is:\n[f{fold:02d}] [{best_iter}]\t{mets2str(acc_t, acc_v, loss_t, loss_v)}"
    logger.info(s)


class BaseModel():
    def __init__(self, merge, models_path):
        merge = self.preproc(merge)
        
        self.train_f = merge.loc[merge["split_type"]=="train"]["fold"]
        self.train_x_common = merge.loc[merge["split_type"]=="train"].drop(["y", "split_type", "fold"], axis=1)
        self.train_y = merge.loc[merge["split_type"]=="train"]["y"]
        self.test_x_common = merge.loc[merge["split_type"]=="test"].drop(["y", "split_type", "fold"], axis=1)  

        self.mets = []
        self.importance = defaultdict(list)

        self.preds_test_all = []

        self.preds_valid_all = self.train_f.copy().to_frame()
        self.preds_valid_all["pred"] = np.nan

        self.preds_train_all = [self.train_f.copy().to_frame()]

        # lgbmの中間結果
        self.evals_df = []

        self.models_path = models_path


    def get_fold_data(self, fold):
        logger.info(f"Start f{fold:02d}")
        
        merge_1fold = self.merge_1fold
        del self.merge_1fold
        
        tdx = self.train_f[self.train_f != fold].index
        vdx = self.train_f[self.train_f == fold].index

        # 対象Foldの特徴量を追加（for target enc）
        if merge_1fold is None:
            train_x = self.train_x_common.copy()
            test_x = self.test_x_common.copy()
        else:
            merge_1fold = self.preproc(merge_1fold)

            train_num = len(self.train_x_common.index)
            # メモリ消費削減のため。順番に並んでること前提。fancy indexじゃなくてスライスだとshallow copyになるはず。
            train_x_fold_specific = merge_1fold.iloc[:train_num,:]
            test_x_fold_specific = merge_1fold.iloc[train_num:,:]
            del merge_1fold
            gc.collect()

            train_x = pd.concat([self.train_x_common, train_x_fold_specific], axis=1)
            del train_x_fold_specific
            gc.collect()

            test_x = pd.concat([self.test_x_common, test_x_fold_specific], axis=1)
            del test_x_fold_specific
            gc.collect()

        X_train, X_valid, y_train, y_valid = train_x.loc[tdx, :], train_x.loc[vdx, :], self.train_y.loc[tdx], self.train_y.loc[vdx]

        if self.selected_cols is not None:  # 順番keep
            X_train = X_train[self.selected_cols]
            X_valid = X_valid[self.selected_cols]
            test_x = test_x[self.selected_cols]

        logger.info(f"Finish f{fold:02d}")
        return X_train, X_valid, y_train, y_valid, test_x, vdx, tdx

    def preproc(self, merge):
        return merge

    def train_1fold(self, fold, params, params_custom):
        pass

    def save(self, is_finish=True, is_train_folds_full=True):
        models_path = self.models_path

        # 変換
        mets = pd.DataFrame(self.mets, \
                columns=["fold", "acc_train", "acc_valid", "logloss_train", "logloss_valid", "iteration"])

        importance = dict()
        for k in self.importance.keys():
            importance[k] = pd.concat(self.importance[k], axis=1)
            importance[k].index.name = "feature"
        
        if len(self.evals_df) > 0:
            eval_df = pd.concat(self.evals_df, axis=1)
            eval_df.to_csv(models_path + "/evals.csv", index=False)

        preds_final = pd.DataFrame(self.preds_test_all).mean().values
        submission = np.round(preds_final)

        # Display #################
        if is_finish:
            print('')
            print('#'*50)
            print("mets mean")
            display(mets.mean().to_frame().T)

            print("mets each fold")
            display(mets)

            if len(importance) > 0:
                if "gain" in importance:
                    k = "gain"
                else:
                    k = list(importance.keys())[0]
                print(f"importance {k} top_n")
                display(importance[k].mean(axis=1).sort_values(ascending=False).to_frame().iloc[:200])
        
        # Save ###############
        
        # 提出物
        if is_finish:
            pd.DataFrame({"id": range(len(submission)), "y": submission}).to_csv(models_path + "/submission.csv", index=False)
        
            # 予測結果
            np.savetxt(models_path + "/preds_final.csv", preds_final)

        if self.preds_train_all is not None:
            pd.concat(self.preds_train_all, axis=1, sort=False).to_csv(models_path + "/preds_train.csv", index=True)
        if is_finish and is_train_folds_full:
            assert self.preds_valid_all.isnull().any().any() == False
        self.preds_valid_all.to_csv(models_path + "/preds_valid.csv", index=True)
        pd.DataFrame(self.preds_test_all).T.to_csv(models_path + "/preds_test.csv", index=True)
        
        # 評価結果
        mets.to_csv(models_path + "/mets.csv", index=False)

        if is_finish:
            mets.mean().to_csv(models_path + "/mets_mean.csv")
        
        for k in importance.keys():
            importance[k].reset_index().to_csv(models_path + f"/importance_{k}.csv", index=False)
        
        return None  


class LgbmModel(BaseModel):
    def train_1fold(self, fold, params, params_custom):
        X_train, X_valid, y_train, y_valid, X_test, vdx, tdx = self.get_fold_data(fold)
        
        if fold == 0:
            X_train.dtypes.to_csv(self.models_path + "/dtypes.csv")
        logger.info(f"X_train.shape = {X_train.shape} f{fold:02d}")

        mprof_timestamp(f"lgb_dataset_f{fold}")
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_valid, y_valid)

        feval = build_feval(params_custom)

        # 学習の実行
        evals = dict()
        params2 = copy.deepcopy(params)
        callbacks = [log_evaluation(logger, period=10)]
        if params2["seed"] is not None:
            params2["seed"] = params2["seed"] + fold
            logger.info(f"Set lgbm train seed = {params2['seed']}")

        logger.info(f"Start train f{fold:02d}")
        mprof_timestamp(f"lgb_train_f{fold}")
        model = lgb.train(params2, lgb_train,
                            valid_names=["train", "valid"], valid_sets=[lgb_train, lgb_valid],
                            verbose_eval=100, evals_result=evals, feval=feval, callbacks=callbacks, 
                            keep_training_booster=True)

        logger.info(f"current_iteration={model.current_iteration()}")
        logger.info(f"best_iteration={model.best_iteration}")

        mprof_timestamp(f"lgb_postproc_f{fold}")
        model.save_model(self.models_path + f"/model-lgbm-f{fold:02d}.txt", num_iteration=model.best_iteration)
        model.save_model(self.models_path + f"/model-lgbm-last-f{fold:02d}.txt", num_iteration=-1)

        evals_df = pd.DataFrame({
            f"logloss_train_f{fold:02d}":evals["train"]['logloss'],
            f"accuracy_train_f{fold:02d}":evals["train"]['accuracy'],
            f"logloss_valid_f{fold:02d}":evals['valid']['logloss'],
            f"accuracy_valid_f{fold:02d}":evals['valid']['accuracy']
        })
        self.evals_df.append(evals_df)

        # 予測値の保存
        preds_valid = model.predict(X_valid, num_iteration=model.best_iteration)
        self.preds_valid_all.loc[vdx, "pred"] = preds_valid

        preds_train = model.predict(X_train, num_iteration=model.best_iteration)
        self.preds_train_all.append(pd.DataFrame({fold:preds_train}, index=tdx))

        preds_test = model.predict(X_test, num_iteration=model.best_iteration)
        self.preds_test_all.append(preds_test)

        # 性能指標の保存
        ms = [fold, model.best_score["train"]["accuracy"], model.best_score["valid"]["accuracy"], 
                model.best_score["train"]["logloss"], model.best_score["valid"]["logloss"], model.best_iteration]
        self.mets.append(ms)
        show_mets(*ms)
        
        for it in ["gain", "split"]:
            imp = pd.Series(model.feature_importance(importance_type=it, iteration=model.best_iteration), 
                    index=model.feature_name())
            imp.name = fold
            imp.index.name = "feature"
            self.importance[it].append(imp)

    def preproc(self, merge):
        #merge = _merge.copy()

        cols_weapon = [x for x in merge.columns if re.search("^(?=([AB][1234]-weapon))(?!.*-(tar|freq)enc)", x)]  # 列ごとのweapon
        for c in cols_weapon:
            merge[c] = merge[c].cat.codes

        cols = [x for x in merge.columns if re.fullmatch("[AB][1234]-rank", x)]
        if len(cols) > 0:
            merge = merge.drop(cols, axis=1)  # []はerrorにならない

        return merge     


class CatBoostModel(BaseModel):
    def train_1fold(self, fold, params, params_custom):
        X_train, X_valid, y_train, y_valid, X_test, vdx, tdx = self.get_fold_data(fold)

        cat_feature_idx = []
        for i, c in enumerate(X_train):
            if not is_numeric_dtype(X_train[c]):
                cat_feature_idx.append(i)

        if fold == 0:
            X_train.dtypes.to_csv(self.models_path + "/dtypes.csv")
            logger.info(f"X_train.shape = {X_train.shape}")

        params2 = copy.deepcopy(params)
        if params2["random_seed"] is not None:
            params2["random_seed"] = params2["random_seed"] + fold
            logger.info(f"Set catboost train random_seed = {params2['random_seed']}")

        model = CatBoostClassifier(**params2)

        model.fit(
            X_train, y_train,
            cat_features=cat_feature_idx,
            eval_set=(X_valid, y_valid)
        )

        model.save_model(self.models_path + f'/model-catboost-f{fold:02d}.bin')
        util.dump_json(model.get_all_params(), self.models_path + "/params.json")

        evals = model.get_evals_result()
        evals_df = pd.DataFrame({
            f"logloss_train_f{fold:02d}":evals["learn"]['Logloss'],
            f"accuracy_train_f{fold:02d}":evals["learn"]['Accuracy'],
            f"logloss_valid_f{fold:02d}":evals['validation']['Logloss'],
            f"accuracy_valid_f{fold:02d}":evals['validation']['Accuracy']
        })
        self.evals_df.append(evals_df)

        preds_valid = model.predict_proba(X_valid)[:,1]
        logger.info(f"len(vdx)={len(vdx)} len(preds_valid)={len(preds_valid)}")
        self.preds_valid_all.loc[vdx, "pred"] = preds_valid

        preds_train = model.predict_proba(X_train)[:,1]
        self.preds_train_all.append(pd.DataFrame({fold:preds_train}, index=tdx))

        preds_test = model.predict_proba(X_test)[:,1]
        self.preds_test_all.append(preds_test)

        acc_valid = accuracy_score(y_valid, np.round(preds_valid))
        acc_train = accuracy_score(y_train, np.round(preds_train))
        logloss_valid = log_loss(y_valid, preds_valid)
        logloss_train = log_loss(y_train, preds_train)

        ms = [fold, acc_train, acc_valid, logloss_train, logloss_valid, model.get_best_iteration()]
        self.mets.append(ms)
        show_mets(*ms)

        for it in ["FeatureImportance"]:
            imp = pd.Series(model.get_feature_importance(type=it), index=X_train.columns)
            imp.name = fold
            imp.index.name = "feature"
            self.importance[it].append(imp)

    def preproc(self, _merge):
        merge = _merge.copy()

        for n in merge.columns:
            if not is_numeric_dtype(merge[n]): # str, cat
                if is_categorical_dtype(merge[n]):
                    merge[n] = merge[n].astype(str)
                merge[n].fillna("NA", inplace=True)
            elif (n in ["A1-uid"]): # 数値型は置き換え or 追加
                merge[n] = merge[n].astype(str).fillna("NA")
            elif (n in ["A1-level_bin"]) \
                    or (re.match("period_", n) is not None) \
                    or (re.fullmatch(r"[AB][1234]-level_bin_q\d+", n) is not None) \
                    or (re.fullmatch("[AB][1234]-level", n) is not None):
                merge[n + "_str"] = merge[n].astype(str).fillna("NA")

        return merge


class MeanModel(BaseModel):
    def train_1fold(self, fold, params, params_custom):
        X_train, X_valid, y_train, y_valid, X_test, vdx, tdx = self.get_fold_data(fold)

        if fold == 0:
            X_train.dtypes.to_csv(self.models_path + "/dtypes.csv")
            logger.info(f"X_train.shape = {X_train.shape}")

        preds_valid = X_valid.mean(axis=1)
        self.preds_valid_all.loc[vdx, "pred"] = preds_valid

        preds_train = X_train.mean(axis=1)
        self.preds_train_all.append(pd.DataFrame({fold:preds_train}, index=tdx))

        preds_test = X_test.mean(axis=1)
        self.preds_test_all.append(preds_test)

        acc_valid = accuracy_score(y_valid, np.round(preds_valid))
        acc_train = accuracy_score(y_train, np.round(preds_train))
        logloss_valid = log_loss(y_valid, preds_valid)
        logloss_train = log_loss(y_train, preds_train)

        ms = [fold, acc_train, acc_valid, logloss_train, logloss_valid, None]
        self.mets.append(ms)
        show_mets(*ms)


    def preproc(self, merge):
        return merge


class LinearModel(BaseModel):
    def train_1fold(self, fold, params, params_custom):
        X_train, X_valid, y_train, y_valid, X_test, vdx, tdx = self.get_fold_data(fold)

        if fold == 0:
            X_train.dtypes.to_csv(self.models_path + "/dtypes.csv")
            logger.info(f"X_train.shape = {X_train.shape}")

        params2 = copy.deepcopy(params)
        if params2["random_state"] is not None:
            params2["random_state"] = params2["random_state"] + fold
            logger.info(f"Set {self.model_type} train random_state = {params2['random_state']}")

        model = self.model_class(**params2)
        model.fit(X_train, y_train)

        joblib.dump(model, self.models_path + f'/model-{self.model_type}-f{fold:02d}.joblib')
        util.dump_json({
            "coef":list(model.coef_[0]), 
            "intercept":model.intercept_[0],
            "coef_name":list(X_train.columns)
            }, 
            self.models_path + f'/model-{self.model_type}-f{fold:02d}.joblib'
            )

        preds_valid = self.predict_proba(model, X_valid)
        self.preds_valid_all.loc[vdx, "pred"] = preds_valid

        preds_train = self.predict_proba(model, X_train)
        self.preds_train_all.append(pd.DataFrame({fold:preds_train}, index=tdx))

        preds_test = self.predict_proba(model, X_test)
        self.preds_test_all.append(preds_test)

        acc_valid = accuracy_score(y_valid, np.round(preds_valid))
        acc_train = accuracy_score(y_train, np.round(preds_train))
        logloss_valid = log_loss(y_valid, preds_valid)
        logloss_train = log_loss(y_train, preds_train)

        ms = [fold, acc_train, acc_valid, logloss_train, logloss_valid, None]
        self.mets.append(ms)
        show_mets(*ms)

        imp = pd.Series(model.coef_[0], index=X_train.columns)
        imp.name = fold
        imp.index.name = "feature"
        self.importance["coef_abs"].append(imp.abs())
        self.importance["coef"].append(imp)

    def preproc(self, merge):
        #merge = _merge.copy()

        cols_exc = ["y", "id", "index", "fold", "split_type"]
        cols_exc_2 = []

        remove_cols = []
        merge_onehot = []
        for n, t in merge.dtypes.to_dict().items():
            if n in cols_exc:
                cols_exc_2.append(n)
            elif re.match("(DiffType-)|(.*-cnt-)", n):
                remove_cols += [n]
            elif n in ["mode", "lobby-mode", "stage",\
                 "A1-level_bin", "period_month", "period_day", "period_weekday", "period_hour", "period_2W"]: #"A1-uid",
                # カテゴリカル特徴で残すもの
                dm = pd.get_dummies(merge[n], prefix=f"{n}-onehot")
                merge_onehot.append(dm)
                remove_cols += [n]               
            elif is_categorical_dtype(merge[n]) or is_string_dtype(merge[n]):
                # 他のカテゴリカル特徴は削除
                remove_cols += [n]  
            elif is_numeric_dtype(merge[n]):
                merge[n] = scipy.stats.zscore(merge[n], nan_policy="omit")
                merge[n].fillna(0, inplace=True)
            else:
                assert False, (n, t)

        merge.drop(remove_cols, axis=1, inplace=True)
        merge = pd.concat([merge] + merge_onehot, axis=1)

        m = merge.drop(cols_exc_2, axis=1)
        assert m.isnull().any().any() == False
        assert m.select_dtypes(exclude='number').shape[1]==0, m.select_dtypes(exclude='number').dtypes

        return merge


class RidgeModel(LinearModel):
    def __init__(self, merge, models_path):
        super().__init__(merge, models_path)
        self.model_type = "ridge"
        self.model_class = RidgeClassifier

    def predict_proba(self, model, dat):
        p = model.decision_function(dat)
        p = np.exp(p) / (1 + np.exp(p))
        return p


class LogisticModel(LinearModel):
    def __init__(self, merge, models_path):
        super().__init__(merge, models_path)
        self.model_type = "logistic"
        self.model_class = LogisticRegression  

    def predict_proba(self, model, dat):
        p = model.predict_proba(dat)[:,1]
        return p


class RFModel(BaseModel):
    def train_1fold(self, fold, params, params_custom):
        X_train, X_valid, y_train, y_valid, X_test, vdx, tdx = self.get_fold_data(fold)
        
        if fold == 0:
            X_train.dtypes.to_csv(self.models_path + "/dtypes.csv")
            logger.info(f"X_train.shape = {X_train.shape}")

        params2 = copy.deepcopy(params)
        if params2["random_state"] is not None:
            params2["random_state"] = params2["random_state"] + fold
            logger.info(f"Set RF train random_state = {params2['random_state']}")

        model = RandomForestClassifier(**params2)
        model.fit(X_train, y_train)
        joblib.dump(model, self.models_path + f'/model-rf-f{fold:02d}.joblib')

        preds_valid = model.predict_proba(X_valid)[:,1]
        self.preds_valid_all.loc[vdx, "pred"] = preds_valid

        preds_train = model.predict_proba(X_train)[:,1]
        self.preds_train_all.append(pd.DataFrame({fold:preds_train}, index=tdx))

        preds_test = model.predict_proba(X_test)[:,1]
        self.preds_test_all.append(preds_test)

        acc_valid = accuracy_score(y_valid, np.round(preds_valid))
        acc_train = accuracy_score(y_train, np.round(preds_train))
        logloss_valid = log_loss(y_valid, preds_valid)
        logloss_train = log_loss(y_train, preds_train)

        ms = [fold, acc_train, acc_valid, logloss_train, logloss_valid, None]
        self.mets.append(ms)
        show_mets(*ms)

        imp = pd.Series(model.feature_importances_, index=X_train.columns)
        imp.name = fold
        imp.index.name = "feature"
        self.importance["impurity"].append(imp.abs())


    def preproc(self, merge):
        for n in merge.columns:
            if is_categorical_dtype(merge[n]):
                merge[n] = merge[n].cat.codes

        cols = [x for x in merge.columns if re.fullmatch("[AB][1234]-rank", x)]
        merge = merge.drop(cols, axis=1)  # []はerrorにならない

        cols = [x for x in merge.columns if x not in ["y", "fold"]]
        merge[cols] = merge[cols].fillna(-999)

        return merge  


class XgbModel(BaseModel):
    def train_1fold(self, fold, params, params_custom):
        X_train, X_valid, y_train, y_valid, X_test, vdx, tdx = self.get_fold_data(fold)

        if fold == 0:
            X_train.dtypes.to_csv(self.models_path + "/dtypes.csv")
            logger.info(f"X_train.shape = {X_train.shape}")

        params2 = copy.deepcopy(params)
        if params2["seed"] is not None:
            params2["seed"] = params2["seed"] + fold
            logger.info(f"Set Xgb train seed = {params2['seed']}")

        xy_train = xgb.DMatrix(X_train, label=y_train)
        xy_valid = xgb.DMatrix(X_valid, label=y_valid)
        xgb_test = xgb.DMatrix(X_test)

        evals = [(xy_train, 'train'), (xy_valid, 'valid')]

        evals_result  = dict()
        model = xgb.train(params, xy_train, num_boost_round=params_custom['num_boost_round'], 
            evals=evals, evals_result=evals_result, early_stopping_rounds=params_custom['early_stopping_rounds'],
            verbose_eval=10)

        model.save_model(self.models_path + f'/model-xgb-f{fold:02d}.bin')

        evals_df = pd.DataFrame({
            f"logloss_train_f{fold:02d}":evals_result["train"]['logloss'],
            #f"accuracy_train_f{fold:02d}":evals_result["train"]['Accuracy'],
            f"logloss_valid_f{fold:02d}":evals_result['valid']['logloss'],
            #f"accuracy_valid_f{fold:02d}":evals_result['valid']['Accuracy']
        })
        self.evals_df.append(evals_df)

        preds_valid = model.predict(xy_valid)
        self.preds_valid_all.loc[vdx, "pred"] = preds_valid

        preds_train = model.predict(xy_train)
        self.preds_train_all.append(pd.DataFrame({fold:preds_train}, index=tdx))

        preds_test = model.predict(xgb_test)
        self.preds_test_all.append(preds_test)

        acc_valid = accuracy_score(y_valid, np.round(preds_valid))
        acc_train = accuracy_score(y_train, np.round(preds_train))
        logloss_valid = log_loss(y_valid, preds_valid)
        logloss_train = log_loss(y_train, preds_train)

        ms = [fold, acc_train, acc_valid, logloss_train, logloss_valid, model.best_iteration]
        self.mets.append(ms)
        show_mets(*ms)

        for it in ["gain", "weight", "cover", "total_gain", "total_cover"]:
            imp = pd.Series(model.get_score(importance_type=it))
            imp.name = fold
            imp.index.name = "feature"
            self.importance[it].append(imp)

    def preproc(self, _merge):
        merge = _merge.copy()

        for n in merge.columns:
            if is_categorical_dtype(merge[n]):
                merge[n] = merge[n].cat.codes

        cols = [x for x in merge.columns if re.fullmatch("[AB][1234]-rank", x)]
        merge = merge.drop(cols, axis=1)  # []はerrorにならない

        return merge


class MLPNet(nn.Module):
    def __init__(self, dim_num, layer_num, drop_out_rate):
        super(MLPNet, self).__init__()
        fcs = []
        for i in range(layer_num):
            in_dim = dim_num//(2**i)
            if i+1 == layer_num:
                out_dim = 1
            else:
                out_dim = dim_num//(2**(i+1))
            fcs.append(nn.Linear(in_dim, out_dim))
        self.fcs = nn.ModuleList(fcs)
        self.dropouts = [nn.Dropout2d(drop_out_rate) for i in range(layer_num-1)]
        
    def forward(self, x):
        x = x["num"]
        for i in range(len(self.fcs)-1):
            x = self.dropouts[i](F.relu(self.fcs[i](x)))
        x = self.fcs[-1](x)
        return x


class MLPModel(BaseModel):
    def __init__(self, merge, models_path):
        super().__init__(merge, models_path)
        self.device = torch.device("cuda")

    @staticmethod
    def calc_mets(mets):
        mets = pd.DataFrame(mets)
        loss = (mets[0]*mets[2]).sum()/mets[2].sum()
        acc = (mets[1]*mets[2]).sum()/mets[2].sum()
        return loss, acc

    @staticmethod
    def batch2input(batch, device, keys):
        xs = {}
        for i, key in enumerate(keys):
            xs[key] = batch[i].to(device)
        return xs

    def evaluate(self, model, dataloader, has_y):
        mets = []
        preds_all = []
        with torch.no_grad():
            model.eval()
            for batch in tqdm(dataloader, desc="eval", leave=False):
                xs = MLPModel.batch2input(batch, self.device, self.batch_keys)
                logits = model(xs).view(-1)
                preds = nn.Sigmoid()(logits).cpu().detach().numpy()
                preds_all.append(preds)
                if has_y:
                    y = batch[-1].to(self.device)
                    loss = BCEWithLogitsLoss()(logits, y)
                    acc = accuracy_score(y.cpu().detach().numpy(), np.round(preds))
                    mets.append((loss.item(), acc, len(y)))
        
        preds_all = np.hstack(preds_all)
        loss, acc = None, None
        if has_y:
            loss, acc = MLPModel.calc_mets(mets)
        return preds_all, loss, acc

    def train_1fold(self, fold, params, params_custom):
        x_train, x_valid, y_train, y_valid, x_test, vdx, _ = self.get_fold_data(fold)
        
        batch_size = params["batch_size"]  # 64
        layer_num = params["layer_num"]  # 8
        drop_out_rate = params["drop_out_rate"]  # 0.2

        if fold == 0:
            x_train.dtypes.to_csv(self.models_path + "/dtypes.csv")
            logger.info(f"X_train.shape = {x_train.shape}")

        model = MLPNet(x_train.shape[1], layer_num, drop_out_rate).to(self.device)

        x_train = {"num":x_train.values}
        x_valid = {"num":x_valid.values}
        x_test = {"num":x_test.values}
        y_train = y_train.values
        y_valid = y_valid.values

        self.batch_keys = list(x_train.keys())

        dataset = TensorDataset(*([torch.tensor(x) for x in x_train.values()] + [torch.tensor(y_train)]))
        dataset_valid = TensorDataset(*([torch.tensor(x) for x in x_valid.values()] + [torch.tensor(y_valid)]))
        dataset_test = TensorDataset(*([torch.tensor(x) for x in x_test.values()]))

        dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size)
        dataloader_valid = torch.utils.data.DataLoader(dataset_valid, shuffle=False, batch_size=batch_size)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, shuffle=False, batch_size=batch_size)

        optimizer = torch.optim.Adam(model.parameters())
    
        evals = []
        preds_valid_all = []
        preds_test_all = []
        for epoch in range(10):
            mets = []
            st = time.time()
            for batch in tqdm(dataloader, desc="train", leave=False):
                model.train()
                xs = MLPModel.batch2input(batch, self.device, self.batch_keys)
                y = batch[-1].to(self.device)
                optimizer.zero_grad()
                logits = model(xs).view(-1)
                loss = BCEWithLogitsLoss()(logits, y)
                loss.backward()
                optimizer.step()

                preds = nn.Sigmoid()(logits).cpu().detach().numpy()
                acc = accuracy_score(y.cpu().detach().numpy(), np.round(preds))
                mets.append((loss.item(), acc, len(y)))

            loss_tr, acc_tr = MLPModel.calc_mets(mets)

            preds_valid, loss_vl, acc_vl = self.evaluate(model, dataloader_valid, True)
            preds_valid_all.append(preds_valid)
            
            preds_test, *_ = self.evaluate(model, dataloader_test, False)
            preds_test_all.append(preds_test)

            logger.info(f"\n[{epoch:02d}] " + mets2str(acc_tr, acc_vl, loss_tr, loss_vl) + f" {int(time.time()-st)}(sec)")
            evals.append([epoch, loss_tr, acc_tr, loss_vl, acc_vl])

        evals = pd.DataFrame(evals, columns=["iter", "logloss_train", "accuracy_train", "logloss_valid", "accuracy_valid"])
        best_eval = evals.loc[evals["logloss_valid"].idxmin()]

        ev = evals.drop("iter", axis=1)
        ev.columns = [c + f"_f{fold:02d}" for c in ev.columns]
        self.evals_df.append(ev)

        # 予測値の保存
        it = int(best_eval["iter"])
        self.preds_valid_all.loc[vdx, "pred"] = preds_valid_all[it]
        self.preds_train_all = None
        self.preds_test_all.append(preds_test_all[it])

        # 性能指標の保存
        ms = [fold, best_eval["accuracy_train"], best_eval["accuracy_valid"], 
                best_eval["logloss_train"], best_eval["logloss_valid"], it]
        self.mets.append(ms)
        show_mets(*ms)

    def preproc(self, merge):
        """Ridgeとほぼ同じだが、np.float32変換忘れずに！"""
        #merge = _merge.copy()

        cols_exc = ["y", "id", "index", "fold", "split_type"]
        cols_exc_2 = []

        remove_cols = []
        merge_onehot = []
        for n, t in merge.dtypes.to_dict().items():
            if n in cols_exc:
                cols_exc_2.append(n)
            elif re.match("(DiffType-)|(.*-cnt-)", n):
                remove_cols += [n]
            elif n in ["mode", "lobby-mode", "stage",\
                 "A1-uid", "A1-level_bin", "period_month", "period_day", "period_weekday", "period_hour", "period_2W"]: #"A1-uid",
                # カテゴリカル特徴で残すもの
                dm = pd.get_dummies(merge[n], prefix=f"{n}-onehot").astype(np.float32)
                merge_onehot.append(dm)
                remove_cols += [n]               
            elif is_categorical_dtype(merge[n]) or is_string_dtype(merge[n]):
                # 他のカテゴリカル特徴は削除
                remove_cols += [n]  
            elif is_numeric_dtype(merge[n]):
                merge[n] = scipy.stats.zscore(merge[n], nan_policy="omit")
                merge[n].fillna(0, inplace=True)
                merge[n] = merge[n].astype(np.float32)
            else:
                assert False, (n, t)

        merge.drop(remove_cols, axis=1, inplace=True)
        merge = pd.concat([merge] + merge_onehot, axis=1)

        m = merge.drop(cols_exc_2, axis=1)
        assert m.isnull().any().any() == False
        assert m.select_dtypes(exclude='number').shape[1]==0, m.select_dtypes(exclude='number').dtypes

        return merge
