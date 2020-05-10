# Packages
import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import functools
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sn

import pandas as pd      
import pyspark.sql.functions as F

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import (SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC,
                                    KMeansSMOTE)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import xgboost as xgb

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Options
pd.set_option('float_format', '{:.0f}'.format)
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
np.set_printoptions(precision=2)


def save_parquet_file(DF_name, file_path, file_name):
    _ = pa.Table.from_pandas(DF_name)
    pq.write_table(_, file_path + "\\" + file_name)

def empty(df):
    cols = [cols for cols in df.columns]
    _, __ = df.shape
    df.drop(cols, axis=1, inplace=True)
    df.drop([i for i in range(_)], axis=0, inplace=True)



def do_twice(func):
    @functools.wraps(func)
    def wrapper_do_twice(*args, **kwargs):
        func(*args, **kwargs)
        return func(*args, **kwargs)
    return wrapper_do_twice

def debug(func):
    """
    Print the function signature and return value
    """
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]  # 1
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()] #2
        signature = ",".join(args_repr + kwargs_repr) # 3
        print(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}") # 4
            
        return value
    
    return wrapper_debug


def timer(func):
    """
    Print the runtime of the decorated function
    """
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter() # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter() # 2
        run_time = end_time - start_time # 3
        
        print(f"Finished {func.__name__!r} in {run_time: .4f} secs")
        
        return value
    
    return wrapper_timer

@timer
def distrib_numctr(df):
        df["NUMCTR_INS"] = 0

        dict_contracts = {}

        for i in df.index:
                j = len(df) - i - 1
                if  df.at[j, "NUMCTR"] != 0:
                        ctr = df.at[j, "NUMCTR"]
                        dict_contracts[ctr] = [j]
                        k = 1
                        while (j - k) in df.index and \
                        df.at[(j - k), "NOMGUI"] == df.at[(j), "NOMGUI"] and \
                        df.at[j - k, "NUMTIERS1"] == df.at[(j), "NUMTIERS1"] and \
                        df.at[j - k, "MATRIC"] == df.at[(j), "MATRIC"]:
                                dict_contracts[ctr].append(j-k)
                                k += 1
        for key in dict_contracts:
                df.at[dict_contracts[key], "NUMCTR_INS"] = key

def mad(a, axis=None):
    """
    Compute *Median Absolute Deviation* of an array along given axis.
    """

    # Median along given axis, but *keeping* the reduced axis so that
    # result can still broadcast against a.
    med = np.median(a, axis=axis, keepdims=True)
    mad = np.median(np.absolute(a - med), axis=axis)  # MAD along given axis

    return mad

def remove_duplicates_from_list(x):
        """
        x: a list of elements
        """
        return list(dict.fromkeys(x))

def reduce_to_k_dim(M, k=2):
        
        n_iters = 10
        M_reduced = None
        print(f"Running Truncated SVD over {M.shape[0]} records...")

        svd = TruncatedSVD(n_components=k, n_iter=n_iters, random_state=42)
        svd.fit_transform(M)
        print(f"Explained variance ratios: {svd.explained_variance_ratio_}")
        M_reduced = svd.fit_transform(M)
        print("Done.")

        return M_reduced

def reduce_to_k_dim_PCA(M, k=2, title=None, path=None, file_name=None, kernel=None, plot_graph=True):
        
        M_reduced = None
        print(f"Running PCA over {M.shape[0]} records...")
        
        if kernel is not None:
                rbf_pca = KernelPCA(n_components=k,
                                    kernel="rbf",
                                    gamma=0.04,
                                    )
                
                M_reduced = rbf_pca.fit_transform(M)
                cumsum = np.cumsum(rbf_pca.explained_variance_ratio_)
        else:
                pca = PCA(n_components=k, random_state=42)
                M_reduced = pca.fit_transform(M)
                cumsum = np.cumsum(pca.explained_variance_ratio_)

        print(f"Cumsum:{cumsum}")
        d55 = np.argmax(cumsum >= 0.55) + 1
        d70 = np.argmax(cumsum >= 0.70) + 1
        d85 = np.argmax(cumsum >= 0.85) + 1
        d90 = np.argmax(cumsum >= 0.90) + 1
        print(f"Initial number of dimensions:{M.shape[1]}")
        print(f"Number of principal components explaining 55% of the variance: {d55}")
        print(f"Number of principal components explaining 70% of the variance: {d70}")
        print(f"Number of principal components explaining 85% of the variance: {d85}")
        print(f"Number of principal components explaining 90% of the variance: {d90}")
        print(f"Explained variance ratios: {pca.explained_variance_ratio_}")
        print("Done.")
        if plot_graph is not False:
                plt.figure(figsize=(8, 3.5))
                plt.plot(range(1, k+1), cumsum, "bo-")
                plt.xlabel("Dimensions", fontsize=14)
                plt.ylabel("Explained Variance Ratio", fontsize=14)
                # plt.annotate("Elbow",
                #         xy=(3, inertias[2]),
                #         xytext=(0.55, 0.55),
                #         textcoords="figure fraction",
                #         fontsize=16,
                #         arrowprops=dict(facecolor="black", shrink=0.1)
                #         )
                # plt.axis([1, (k + 0.5), 0, 1])
                plt.xticks(range(1, k+1))
                if title is not None:
                        plt.title(title)
                if (path is not None) and (file_name is not None):
                        plt.savefig(path + file_name)
                plt.show()

        return M_reduced


def plot_clusters(X, y=None):
        plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
        plt.xlabel("$x_1$", fontsize=14)
        plt.ylabel("$x_2$", fontsize=14, rotation=0)

def plot_data(X):
        plt.plot(X[:, 0], X[:, 1], "k", markersize=2)

def plot_centroids(centroids, weights=None, circle_color="w", cross_color="k"):
        if weights is not None:
                centroids = centroids[weights > weights.max() / 10]
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    marker="o", s=30, linewidths=8,
                    color=circle_color, zorder=10, alpha=0.9)
        plt.scatter(centroids[:,0], centroids[:, 1],
                    marker="x", s=50, linewidths=50,
                    color=cross_color, zorder=11, alpha=1)

def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
        mins = X.min(axis=0) - 0.1
        maxs = X.max(axis=0) + 0.1
        xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                             np.linspace(mins[1], maxs[1], resolution))
        Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                     cmap="Pastel2")

        plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                    linewidths=1, colors="k")

        plot_clusters(X)
        if show_centroids:
                plot_centroids(clusterer.cluster_centers_)

        if show_xlabels:
                plt.xlabel("$x_1$", fontsize=14)
        else:
                plt.tick_params(labelbottom=False)
        if show_ylabels:
                plt.ylabel("$x_2$", fontsize=14, rotation=0)
        else:
                plt.tick_params(labelleft=False)


def save_fig(path, fig_id, tight_layout=True):
        print(f"Saving figure {fig_id}")
        if tight_layout:
                plt.tight_layout()
        plt.savefig(path, format="png", dpi=300)

def initialize_table(dtypes):
    columns = [x for x, i in dtypes.items()]
    init_table = pd.DataFrame(columns=columns)
    init_table = init_table.astype(dtype=dtypes)
    return init_table
    
def concat_table_excel(io, dtypes, input_table, usecols=None, infer_datetime_format=False, dayfirst=True):
    columns = [x for x, i in dtypes.items()]
    _temp = pd.read_excel(io=io,
                          header=None,
                          skiprows=1,
                          names=columns,
                          dtype=dtypes,
                          usecols=usecols,
                          infer_datetime_format=infer_datetime_format,
                          dayfirst=dayfirst
                          )

    return pd.concat([input_table, _temp], ignore_index=True)

symlist = ["o", "s", 'D', "+", "x", "*", "p", "v", "-", "^"]
collist = ["blue", "grey", "red", "purple", "orange", "salmon", "black", "fuchsia"]

def plot_2d(data, y=None, w=None, alpha_choice=1, title=None):
        """
        Plot 2D data 
        y: vector for colors
        todo: c for color
              cmap for colormap to be used
              s for radius of each circle, 
        housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
        s=housing["population"]/100, label="population", figsize=(10,7),
        c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
        sharex=False)
        plt.legend()
        save_fig("housing_prices_scatterplot")

        """
        if y is None:
                labels =[""]
                idxbyclass = [range(data.shape[0])]
        else:
                labels = np.unique(y)
                idxbyclass = [np.where(y == labels[i])[0] for i in range(len(labels))]

        for i in range(len(labels)):
                plt.plot(data[idxbyclass[i], 0], data[idxbyclass[i], 1],
                         "+",
                         color=collist[i % len(collist)],
                         ls="None",
                         marker=symlist[i % len(symlist)],
                         label=labels[i]
                         )
        
        plt.ylim([np.min(data[:, 1]), np.max(data[:, 1])])
        plt.xlim([np.min(data[:, 0]), np.max(data[:, 0])])
        mx = np.min(data[:, 0])
        maxx = np.max(data[:, 1])
        if title is not None:
                plt.title(title)
        plt.legend()
        if w is not None:
                plt.plot([mx, mxx],
                         [mx * -w[1] / w[2] - w[0] / w[2], maxx * -w[1] / w[2] - w[0] / w[2]],
                         "g",
                         alpha=alpha_choice
                         )

def create_image_from_data(M,ids, rows, path):
    """
    This function creates heatmap images of given rows in a matrix
    M: matrix
    ids: the list of  identifiers for each row in the matrix
    path: path to save the image file
    rows: list of rows

    Please note that if there exist more than one observation per identifier
    then only the last observation would be taken in to account
    """

    for i in rows:
        minimum_value = np.min(M)
        maximum_value = np.max(M)
        obs_number = i
        sqrt = np.int(np.ceil(np.sqrt(M[obs_number].shape)))
        zdim = sqrt**2 - M[obs_number].shape[0]
        z = np.ones((1, zdim), dtype=M[obs_number].dtype)
        representation = np.concatenate((M[obs_number].reshape(1, -1), z), axis = 1)
        # plt.imshow(representation.reshape(sqrt, sqrt), cmap="viridis")
        # plt.colorbar()
        ax = sns.heatmap(representation.reshape(sqrt, sqrt),
                        # vmin=minimum_value,
                        # vmax=maximum_value,
                        cmap="coolwarm",
                        # cbar=False
                        ).get_figure()
        if ids is not None:
                file_name = str(ids[obs_number]) + ".png"
        ax.savefig(path + file_name)
        ax.clear()
        plt.close(ax)

def plot_dbscan(dbscan, X, size, show_xlabels=True, show_ylabels=True):
    core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_mask[dbscan.core_sample_indices_] = True
    anomalies_mask = dbscan.labels_ == -1
    non_core_mask = ~(core_mask | anomalies_mask)

    cores = dbscan.components_
    anomalies = X[anomalies_mask]
    non_cores = X[non_core_mask]
    
    plt.scatter(cores[:, 0], cores[:, 1],
                c=dbscan.labels_[core_mask], marker='o', s=size, cmap="Paired")
    plt.scatter(cores[:, 0], cores[:, 1], marker='*', s=20, c=dbscan.labels_[core_mask])
    plt.scatter(anomalies[:, 0], anomalies[:, 1],
                c="r", marker="x", s=100)
    plt.scatter(non_cores[:, 0], non_cores[:, 1], c=dbscan.labels_[non_core_mask], marker=".")
    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)
    plt.title("eps={:.2f}, min_samples={}".format(dbscan.eps, dbscan.min_samples), fontsize=14)

def ordinalize(x, cutoffs):
    """
    This function applies the transformation from ESL book to ordinal variables (chapter 14)

    Input:
      x: array
      cutoffs: list

    Output:
      Ordinalized x 
    """
    discrete_x = (np.digitize(x, cutoffs) + 1 - 0.5)/(len(cutoffs) + 1)

    return discrete_x

def display_scores(scores):
        print(f"Scores: {scores}")
        print(f"Mean score: {scores.mean()}")
        print(f"Std: {scores.std()}")

def func_new_X(X_1, X_2):
  return np.hstack((X_1.reshape(-1, 1), X_2.reshape(-1, 1)))

def func_train_test_split_resample_xgboost_CM(X,
                                              y,
                                              X_1_range=False,
                                              X_2_range=False,
                                              depth_X1=2,
                                              depth_X2=2,
                                              output_dir=None):
  

  # Resample
  smote = SMOTE(random_state=42)
  X_train, X_test, y_train, y_test = train_test_split(X,
                                                      y,
                                                      stratify=y,
                                                      test_size=0.20)

  X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

  # Split and preprocess X
  if type(X_1_range) is list:
    X_1 = ordinalize(X_train_resampled[:, 0], X_1_range)
    X_1_test = ordinalize(X_test[:, 0], X_1_range)
    print(f"X_1_range is {X_1_range}")

  if X_1_range == "auto":
    tree_model = DecisionTreeClassifier(max_depth=depth_X1, random_state=42)
    
    tree_model.fit(X_train_resampled[:, 0].reshape(-1, 1), y_train_resampled)
    if output_dir is not None:     
        export_graphviz(tree_model,
                        out_file=output_dir + "tree_X1_depth_"+ depth_X1 +".dot",
                        class_names=([str(y) for y in np.unique(y)]),
                        rounded=True,
                        filled=True
                        )

    predict_proba = tree_model.predict_proba(X_train_resampled[:, 0].reshape(-1, 1))

    df = pd.DataFrame(data=np.hstack((X_train_resampled[:, 0].reshape(-1, 1), 
                                      predict_proba[:, 1].reshape(-1, 1))
                                      ),
                      columns=["X_1", "proba"])

    df = pd.concat([df.groupby(['proba'])["X_1"].min(),
                    df.groupby(['proba'])["X_1"].max()],
                    axis=1)

    my_array = np.unique(df.X_1)
    X_1_range = [np.round(my_array[2*i + 1], 0) for i in range(int(len(my_array)/2))]
    X_1 = ordinalize(X_train_resampled[:, 0], X_1_range)
    X_1_test = ordinalize(X_test[:, 0], X_1_range)
    print(f"X_1_range is {X_1_range}")

  if X_1_range is False:
    X_1 = X_train_resampled[:, 0]
    X_1_test = X_test[:, 0]



  if type(X_2_range) is list:
    X_2 = ordinalize(X_train_resampled[:, 0], X_2_range)
    X_2_test = ordinalize(X_test[:, 0], X_2_range)
    print(f"X_2_range is {X_2_range}")

  if X_2_range == "auto":
    tree_model = DecisionTreeClassifier(max_depth=depth_X1, random_state=42)

    tree_model.fit(X_train_resampled[:, 1].reshape(-1, 1), y_train_resampled)

    if output_dir is not None:     
        export_graphviz(tree_model,
                        out_file=output_dir + "tree_X1_depth_"+ depth_X2 +".dot",
                        class_names=([str(y) for y in np.unique(y)]),
                        rounded=True,
                        filled=True
                        ) 

    predict_proba = tree_model.predict_proba(X_train_resampled[:, 1].reshape(-1, 1))

    df = pd.DataFrame(data=np.hstack((X_train_resampled[:, 1].reshape(-1, 1), 
                                      predict_proba[:, 1].reshape(-1, 1))
                                      ),
                      columns=["X_2", "proba"])

    df = pd.concat([df.groupby(['proba'])["X_2"].min(),
              df.groupby(['proba'])["X_2"].max()],
              axis=1)

    my_array = np.unique(df.X_2)
    X_2_range = [np.round(my_array[2*i + 1], 0) for i in range(int(len(my_array)/2))]
    X_2 = ordinalize(X_train_resampled[:, 1], X_2_range)
    X_2_test = ordinalize(X_test[:, 1], X_2_range)
    print(f"X_2_range is {X_2_range}")

  if X_2_range is False:
    X_2 = X_train_resampled[:, 1]
    X_2_test = X_test[:, 1]



  # Put X together
  X_train_resampled = func_new_X(X_1, X_2)
  X_test = func_new_X(X_1_test, X_2_test)


  # Choose classifier
  clf_xgb = xgb.XGBClassifier()
  
  # Scale
  scaler = MinMaxScaler()
  scaler.fit(X_train_resampled)

  print(f"Running XGBoost on {X_train_resampled.shape[0]} observations...")

  y_test_pred = clf_xgb.fit(scaler.transform(X_train_resampled), y_train_resampled).predict(scaler.transform(X_test))


  return confusion_matrix(y_test, y_test_pred), X_1_range, X_2_range, f1_score(y_test, y_test_pred, average="weighted")

def plot_conf_mat(CM, ax, title="Model"):
  sn.heatmap(CM, annot=True, fmt=".0f", ax=ax, square=True)
  ax.set_title(str(title))
  ax.set_ylabel("True label")
  ax.set_xlabel("Predicted label")