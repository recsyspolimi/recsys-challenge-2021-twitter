# LightGBM optimizer

import optuna
from sklearn.metrics import average_precision_score, log_loss
import xgboost as xgb
import optuna.integration.lightgbm as lgb_sequential
import lightgbm as lgb
import catboost as cb


def tune_lightGBM_naive(X_train, X_val, y_train, y_val, n_trials):
    '''
    Simple tuning of lightGBM using the standard Optuna approach. It basically defines a space of parameters and optuna
    search for the optimal value in that space doing combination of parameters values.

    https://medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b7095e99258
    More info here about naive vs sequential lightGBM

    This directly return the model that maximizes the given metrics (rce and avg_precision)
    '''

    class Objective_lightGBM_naive(object):
        def __init__(self, X_train, X_val, y_train, y_val):
            # Hold this implementation specific arguments as the fields of the class.
            self.X_train, self.X_val, self.y_train, self.y_val = X_train, X_val, y_train, y_val
            self.dtrain = lgb.Dataset(self.X_train, label=self.y_train)

        def calculate_ctr(self, gt):
            positive = len([x for x in gt if x == 1])
            ctr = positive / float(len(gt))
            return ctr

        def compute_rce(self, pred, gt):
            cross_entropy = log_loss(gt, pred)
            data_ctr = self.calculate_ctr(gt)
            strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))])
            return (1.0 - cross_entropy / strawman_cross_entropy) * 100.0

        def __call__(self, trial):
            # Calculate an objective value by using the extra arguments.

            self.param = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
                'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
                'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'feature_pre_filter': False,
                'verbose': 2
            }

            gbm = lgb.train(self.param, self.dtrain)
            preds = gbm.predict(self.X_val)
            # pred_labels = np.rint(preds) #qui arrotonda all'intero più vicino
            rce = self.compute_rce(preds, self.y_val)
            avg_precision = average_precision_score(self.y_val, preds)
            return rce, avg_precision

    # Execute an optimization by using an `Objective` instance.
    study = optuna.create_study(directions=["maximize", "maximize"])
    study.optimize(Objective_lightGBM_naive(X_train, X_val, y_train, y_val), n_trials=n_trials)
    return study.best_trials


def tune_lightGBM_sequential(dtrain, dval):
    '''
    This function tunes the lightGBM method in an incremental way on the following hyper-parameters:
    ``lambda_l1``, ``lambda_l2``, ``num_leaves``, ``feature_fraction``, ``bagging_fraction``,
    ``bagging_freq`` and ``min_child_samples``.

    The code will show both the output of the lightGBM training and the optuna output.

    The "single" steps depending on the num_boost_round or early_stopping_rounds come from the lightGBM actual training.
    Once these steps are completed, the model is evaluated and a new trial begins. Trials come from the optuna library.
    The number of trials for each hyper-parameter tuning has been set empirically inside the library, you can find more here:
    https://optuna.readthedocs.io/en/stable/_modules/optuna/integration/_lightgbm_tuner/optimize.html#LightGBMTuner by searching
    for "n_trials"

    '''

    def calculate_ctr(gt):
        positive = len([x for x in gt if x == 1])
        ctr = positive / float(len(gt))
        return ctr

    def compute_rce(preds, train_data):
        gt = train_data.get_label()
        cross_entropy = log_loss(gt, preds)
        data_ctr = calculate_ctr(gt)
        strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))])
        rce = (1.0 - cross_entropy / strawman_cross_entropy) * 100.0
        return ('rce', rce, True)

    def compute_avg_precision(preds, train_data):
        gt = train_data.get_label()
        avg_precision = average_precision_score(gt, preds)
        return ('avg_precision', avg_precision, True)

    params = {
        "objective": "binary",
        "metric": 'binary_logloss',
        "boosting_type": "gbdt",
        "verbose": 2,
        #"learning_rate" : 0.001,
        "num_threads": -1,
    }

    print('Starting training lightGBM sequential')
    model = lgb_sequential.train(
        params, dtrain, valid_sets=[dtrain, dval], verbose_eval=True, num_boost_round=1000, early_stopping_rounds=50 #feval=[compute_rce, compute_avg_precision]
    )

    return model.params


def tune_XGBoost(X_train, X_val, y_train, y_val, n_trials):
    '''
    Simple tuning of lightGBM using the standard Optuna approach. It basically defines a space of parameters and optuna
    search for the optimal value in that space doing combination of parameters values.

    https://medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b7095e99258
    More info here about naive vs sequential lightGBM

    This directly return the model that maximizes the given metrics (rce and avg_precision)
    '''

    class Objective_XGBoost(object):
        def __init__(self, X_train, X_val, y_train, y_val):
            # Hold this implementation specific arguments as the fields of the class.
            self.X_train, self.X_val, self.y_train, self.y_val = X_train, X_val, y_train, y_val
            self.dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
            self.dvalid = xgb.DMatrix(self.X_val, label=self.y_val)

        def calculate_ctr(self, gt):
            positive = len([x for x in gt if x == 1])
            ctr = positive / float(len(gt))
            return ctr

        def compute_rce(self, pred, gt):
            cross_entropy = log_loss(gt, pred)
            data_ctr = self.calculate_ctr(gt)
            strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))])
            return (1.0 - cross_entropy / strawman_cross_entropy) * 100.0

        def __call__(self, trial):
            # Calculate an objective value by using the extra arguments.

            self.param = {
                "verbosity": 0,
                "objective": "binary:logistic",
                # use exact for small dataset.
                "tree_method": "exact",
                # defines booster, gblinear for linear functions.
                "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
                # L2 regularization weight.
                "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
                # L1 regularization weight.
                "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
                # sampling ratio for training data.
                "subsample": trial.suggest_float("subsample", 0.2, 1.0),
                # sampling according to each tree.
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            }

            if self.param["booster"] in ["gbtree", "dart"]:
                # maximum depth of the tree, signifies complexity of the tree.
                self.param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
                # minimum child weight, larger the term more conservative the tree.
                self.param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
                self.param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
                # defines how selective algorithm is.
                self.param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
                self.param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

            if self.param["booster"] == "dart":
                self.param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
                self.param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
                self.param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
                self.param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

            bst = xgb.train(self.param, self.dtrain)
            preds = bst.predict(self.dvalid)
            # pred_labels = np.rint(preds) #qui arrotonda all'intero più vicino
            rce = self.compute_rce(preds, self.y_val)
            avg_precision = average_precision_score(self.y_val, preds)
            return rce, avg_precision

    # Execute an optimization by using an `Objective` instance.
    study = optuna.create_study(directions=["maximize", "maximize"])
    study.optimize(Objective_XGBoost(X_train, X_val, y_train, y_val), n_trials=n_trials)
    return study.best_trials


def tune_CatBoost(X_train, X_val, y_train, y_val, n_trials):
    '''
    Simple tuning of lightGBM using the standard Optuna approach. It basically defines a space of parameters and optuna
    search for the optimal value in that space doing combination of parameters values.

    https://medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b7095e99258
    More info here about naive vs sequential lightGBM

    This directly return the model that maximizes the given metrics (rce and avg_precision)
    '''

    class Objective_CatBoost(object):
        def __init__(self, X_train, X_val, y_train, y_val):
            # Hold this implementation specific arguments as the fields of the class.
            self.X_train, self.X_val, self.y_train, self.y_val = X_train, X_val, y_train, y_val

        def calculate_ctr(self, gt):
            positive = len([x for x in gt if x == 1])
            ctr = positive / float(len(gt))
            return ctr

        def compute_rce(self, pred, gt):
            cross_entropy = log_loss(gt, pred)
            data_ctr = self.calculate_ctr(gt)
            strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))])
            return (1.0 - cross_entropy / strawman_cross_entropy) * 100.0

        def __call__(self, trial):
            # Calculate an objective value by using the extra arguments.

            self.param = {
                "objective": trial.suggest_categorical("objective", ["Logloss"]),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
                "depth": trial.suggest_int("depth", 1, 12),
                "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
                "bootstrap_type": trial.suggest_categorical(
                    "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
                ),
                "used_ram_limit": "4gb",
            }

            if self.param["bootstrap_type"] == "Bayesian":
                self.param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
            elif self.param["bootstrap_type"] == "Bernoulli":
                self.param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

            gbm = cb.CatBoostClassifier(**self.param)
            gbm.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)], verbose=2, early_stopping_rounds=2)
            print('classes:', gbm.classes_)
            preds = gbm.predict_proba(data=self.X_val)
            print('printing shape of prediction:', preds.shape)
            # pred_labels = np.rint(preds)
            rce = self.compute_rce(preds[:, 1], self.y_val)
            avg_precision = average_precision_score(self.y_val, preds[:, 1])
            return rce, avg_precision

    # Execute an optimization by using an `Objective` instance.
    study = optuna.create_study(directions=["maximize", "maximize"])
    study.optimize(Objective_CatBoost(X_train, X_val, y_train, y_val), n_trials=n_trials)
    return study.best_trials

