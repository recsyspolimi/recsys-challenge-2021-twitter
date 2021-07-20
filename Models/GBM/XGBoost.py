import math
import pickle
from abc import ABC

from dask_ml.xgboost import XGBClassifier
from Utils.Base.RecommenderGBM import RecommenderGBM
from Utils.Eval.Metrics import ComputeMetrics as CoMe, CustomEvalXGBoost


class XGBoost(RecommenderGBM, ABC):
    # ---------------------------------------------------------------------------------------------------
    # n_rounds:      Number of rounds for boosting
    # param:         Parameters of the XGB model
    # kind:          Name of the kind of prediction to print [LIKE, REPLY, RETWEET, RETWEET WITH COMMENT]
    # ---------------------------------------------------------------------------------------------------
    # Not all the parameters are explicitated
    # PARAMETERS DOCUMENTATION:https://xgboost.readthedocs.io/en/latest/parameter.html
    # ---------------------------------------------------------------------------------------------------

    def __init__(self,
                 kind="NO_KIND_GIVEN",
                 verbosity: int = 2,
                 booster: str = 'gbtree',
                 nthread: int = None,
                 objective="binary:logistic",  # outputs the binary classification probability
                 colsample_bytree=1,
                 colsample_bylevel=1,
                 colsample_bynode=1,
                 learning_rate: float = 0.3,
                 max_depth: int = 6,  # Max depth per tree
                 reg_alpha=0,  # L1 regularization
                 reg_lambda=1,  # L2 regularization
                 min_child_weight=1,  # Minimum sum of instance weight (hessian) needed in a child.
                 scale_pos_weight=1,
                 gamma=0,
                 max_delta_step=0,
                 base_score=0.5,
                 subsample=1):

        super(XGBoost, self).__init__(
            name="xgboost_classifier",  # name of the recommender
            kind=kind  # what does it recommend?
        )

        # INPUTS
        self.kind = kind

        # CLASS VARIABLES
        # Model
        self.model = XGBClassifier(
            max_depth=max_depth,
            verbosity=verbosity,
            booster=booster,
            nthread=nthread,
            learning_rate=learning_rate,
            gamma=gamma,
            min_child_weight=min_child_weight,
            max_delta_step=max_delta_step,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            colsample_bynode=colsample_bynode,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            scale_pos_weight=scale_pos_weight,
            base_score=base_score,
            objective=objective,
        )
        # Prediction
        self.Y_pred = None
        # Extension of saved file
        self.ext = ".model"

    # -----------------------------------------------------
    #                    fit(...)
    # -----------------------------------------------------
    # dmat_train:  Training set in DMatrix form.
    # dmat_val:    Validation set in DMatrix form provided
    #              in order to use Early Stopping.
    # -----------------------------------------------------
    # sround_model and batch_model are differentiated
    # in order to avoid overwriting. (Maybe not necessary)
    # -----------------------------------------------------
    # TODO: Redundant code here
    # ------------------------------------------------------
    def fit(self,
            X=None,
            y=None,
            classes=None,
            eval_set=None,
            sample_weight=None,
            sample_weight_eval_set=None,
            eval_metric=None,  # list or callable
            early_stopping_rounds: int = 0):
        self.model.fit(
            X=X,
            y=y,
            classes=classes,
            eval_set=None,
            sample_weight=sample_weight,
            sample_weight_eval_set=sample_weight_eval_set,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds
        )

    # Returns the predictions and evaluates them
    # ---------------------------------------------------------------------------
    #                           evaluate(...)
    # ---------------------------------------------------------------------------
    # X_tst:     Features of the test set
    # Y_tst      Ground truth, target of the test set
    # ---------------------------------------------------------------------------
    #           Works for both for batch and single training
    # ---------------------------------------------------------------------------
    def evaluate(self, X=None):
        # Tries to load X and Y if not directly passed
        if (X is None):
            print("No matrix passed, cannot perform evaluation.")

        if (self.model is None):
            print("No model trained, cannot to perform evaluation.")

        else:
            # Retrieving the predictions
            Y_pred = self.predict(X=X)

            # Declaring the class containing the metrics
            cm = CoMe(Y_pred, X)

            # Evaluating
            prauc = cm.compute_prauc()
            rce = cm.compute_rce()
            # Confusion matrix
            conf = cm.confMatrix()
            # Prediction stats
            max_pred, min_pred, avg = cm.computeStatistics()

            return prauc, rce, conf, max_pred, min_pred, avg

    # This method returns only the predictions
    # -------------------------------------------
    #           get_predictions(...)
    # -------------------------------------------
    # X_tst:     Features of the test set
    # -------------------------------------------
    # As above, but without computing the scores
    # -------------------------------------------
    def predict(self, X=None):
        # Tries to load X and Y if not directly passed
        if (X is None):
            print("No matrix passed, cannot provide predictions.")
            return

        if (self.model is None):
            print("No model trained, cannot perform evaluation.")
            return

        return self.model.predict(X=X)

    # --------------------------
    # This method loads a model
    # -------------------------
    # path: path to the model
    # -------------------------
    def load_model(self, path: str = ''):
        self.model = pickle.load(open(f"{path}", "rb"))

    # --------------------------------------------------
    # Returns/prints the importance of the features
    # -------------------------------------------------
    # verbose:   it also prints the features importance
    # -------------------------------------------------
    def get_feat_importance(self, verbose=False):

        importance = self.model.feature_importances_

        if verbose is True:
            for k, v in importance.items():
                print("{0}:\t{1}".format(k, v))

        return importance

    # -----------------------------------------------------
    #        Get the best iteration with ES
    # -----------------------------------------------------
    def getBestIter(self):
        return self.model.best_iteration
