import lightgbm as lgb

class DaskLightGBM():

  def __init__(self, client, **params):

    '''
    params = {'n_estimators': 50,
              'learning_rate': 0.05,
              'max_depth': 5,
              'random_state': 42,
              'n_jobs': -1,
              'silent': False}
    '''

    self.model = lgb.DaskLGBMClassifier(client=client, **params)


  def fit(self, X_train, y_train, X_val, y_val):
    self.model.fit(X=X_train, y=y_train, eval_set=[(X_val, y_val)], verbose=True)


  def predict(self, X):
    return self.model.predict(X, num_iteration = self.model.best_iteration_)
  

  def evaluate(self, X_test, y_test):
    predictions = self.predict(X_test)
    rce = compute_rce(np.array(y_test).ravel(), predictions)
    average_precision = average_precision_score(np.array(y_test), predictions)
    print(f'RCE : {rce} -- AP {average_precision}')
    return rce, average_precision
  

  def get_best_iter(self):
    return self.model.best_iteration_


  def load_model(self, path: str = '', fname: str = ''):
    from sklearn.externals import joblib
    self.model = joblib.load(f'{path}/{fname}')
    print(f'Model {fname} correctly loaded\n')


  def save_model(self, path: str = '', fname: str = ''):
    from sklearn.externals import joblib
    joblib.dump(self.model, f'{path}/{fname}.pkl')
    print(f'Model correctly saved in {path}/{fname}\n')
  

  def get_feature_importance(self, verbose=False):
    importances = self.model.feature_importances_
    if verbose:
      for i in range(len(importances)):
        print(f'{i}\t{importances[i]}')
    return importances


  def save_feature_importance(self, fname: str = ''):
    import pandas as pd

    feature_imp_split = pd.DataFrame({'Value': self.model.booster_.feature_importance(importance_type="split"),
                                      'Feature': self.model.booster_.feature_name()})
    feature_imp_split.to_csv(fname + '_split.csv')
    
    feature_imp_gain = pd.DataFrame({'Value': self.model.booster_.feature_importance(importance_type="gain"),
                                     'Feature': self.model.booster_.feature_name()})
    feature_imp_gain.to_csv(fname + '_gain.csv')


  def plot_feature_importance(self):
    import matplotlib.pyplot as plt

    lgb.plot_importance(self.model, importance_type="split")
    plt.rcParams["figure.figsize"] = (50, 50)
    plt.title("Feature Importance: SPLIT")
    plt.savefig("fimportance_split.png")

    lgb.plot_importance(model.model, importance_type="gain")
    plt.rcParams["figure.figsize"] = (50, 50)
    plt.title("Feature Importance: GAIN")
    plt.savefig("fimportance_gain.png")


  def plot_tree(self, index):
    import matplotlib.pyplot as plt
    lgb.plot_tree(self.model.booster_, tree_index=index)
    plt.rcParams['figure.figsize'] = (50, 50)
    plt.title(f'Tree {index}')
    plt.savefig(f'tree{index}.png')


  def trees_to_csv(self):
    import pandas as pd
    df = self.model.booster_.trees_to_dataframe()
    df.to_csv('trees.csv')