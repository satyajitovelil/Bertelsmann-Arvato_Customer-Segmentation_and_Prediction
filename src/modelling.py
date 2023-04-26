from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer


def GridSearch_ClassifierCV(model, X, y, params={}, cv=5):
    """
    This function performs grid search cross validation with a given classifier model.

    Parameters:
    -----------
    model: sklearn classifier object
        The classifier model to be used.

    params: dict
        A dictionary of hyperparameters that will be tuned through grid search. Default is an empty dictionary.

    X: {array-like, sparse matrix}
        The training features. Default is the initialized ones.

    y: class labels
        The training labels. Default is the initialized ones.

    Returns:
    --------
    tuple:
        The best score and the hyperparameters that gave the best score.
    """

    evaluator = 'roc_auc'

    search = GridSearchCV(model, cv=cv, scoring=evaluator, n_jobs=-1,
                          param_grid=params
                          )
    search.fit(X, y)

    return search

def plot_elbow(df, k=(2,32), randomState=42):
    '''Creates Elbow Graph for KMeans using KElbowVisualizer'''
    kmeans_yb = KMeans(random_state=randomState, n_init='auto')
    elbow_visualizer = KElbowVisualizer(kmeans_yb, k=k)
    elbow_visualizer.fit(df)
    elbow_visualizer.show();