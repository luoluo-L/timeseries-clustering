# -*- coding: utf-8 -*-
"""
k-means
=======

This example uses :math:`k`-means clustering for time series. Three variants of
the algorithm are available: standard
Euclidean :math:`k`-means, DBA-:math:`k`-means (for DTW Barycenter
Averaging [1])
and Soft-DTW :math:`k`-means [2].

In the figure below, each row corresponds to the result of a different
clustering. In a row, each sub-figure corresponds to a cluster.
It represents the set
of time series from the training set that were assigned to the considered
cluster (in black) as well as the barycenter of the cluster (in red).

A note on pre-processing
~~~~~~~~~~~~~~~~~~~~~~~~

In this example, time series are preprocessed using
`TimeSeriesScalerMeanVariance`. This scaler is such that each output time
series has zero mean and unit variance.
The assumption here is that the range of a given time series is uninformative
and one only wants to compare shapes in an amplitude-invariant manner (when
time series are multivariate, this also rescales all modalities such that there
will not be a single modality responsible for a large part of the variance).
This means that one cannot scale barycenters back to data range because each
time series is scaled independently and there is hence no such thing as an
overall data range.

[1] F. Petitjean, A. Ketterlin & P. Gancarski. A global averaging method \
for dynamic time warping, with applications to clustering. Pattern \
Recognition, Elsevier, 2011, Vol. 44, Num. 3, pp. 678-693
[2] M. Cuturi, M. Blondel "Soft-DTW: a Differentiable Loss Function for \
Time-Series," ICML 2017.
"""

# Author: Romain Tavenard
# License: BSD 3 clause

import numpy
import matplotlib.pyplot as plt

from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, \
    TimeSeriesResampler
import os

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html

# 0 - randomly labeling
# 1 - exact identical and allows random permutation


savefig=False
fig_folder = "num_clusters_figs"
seed = 0
numpy.random.seed(seed)
X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
print(X_train.shape)
#X_train = X_train[y_train < 4]  # Keep first 3 classes
#numpy.random.shuffle(X_train) remove shuffle
#
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
# Make time series shorter
X_train = TimeSeriesResampler(sz=40).fit_transform(X_train)
sz = X_train.shape[1]

num_clusters_array = [2,3,4,5,6,7,8]
dict_kmeans = {}
dict_dba = {}
dict_softdtw = {}

dict_kmeans_ari = {}
dict_dba_ari= {}
dict_softdtw_ari= {}


dict_kmeans_nmi= {}
dict_dba_nmi= {}
dict_softdtw_nmi= {}

for num_cluster in num_clusters_array:
    # Euclidean k-means
    print("Euclidean k-means")
    km = TimeSeriesKMeans(n_clusters=num_cluster, verbose=True, random_state=seed)
    y_pred = km.fit_predict(X_train)
    y_length = y_pred.shape[0]

    plt.figure(figsize=(12,6))
    array_kmeans = []
    array_dba = []
    array_soft_dba = []
    for yi in range(num_cluster):

        ratio_each_cluster = 1.0*y_pred[y_pred==yi].shape[0]/ y_length
        array_kmeans.append(ratio_each_cluster)
        plt.subplot(3, num_cluster, yi + 1)
        for xx in X_train[y_pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.2)
        plt.plot(km.cluster_centers_[yi].ravel(), "r-")
        plt.xlim(0, sz)
        plt.ylim(-4, 4)
        plt.text(0.55, 0.85,'Cluster %d' % (yi + 1) + ': %2.1f%%' % (100*ratio_each_cluster),
                 transform=plt.gca().transAxes)
        if yi == 1:
            plt.title("Euclidean $k$-means")

    kmeans_ari = adjusted_rand_score(y_train, y_pred)
    kmeans_nmi = normalized_mutual_info_score(y_train, y_pred)
    # DBA-k-means
    print("DBA k-means")
    dba_km = TimeSeriesKMeans(n_clusters=num_cluster,
                              n_init=2,
                              metric="dtw",
                              verbose=True,
                              max_iter_barycenter=10,
                              random_state=seed)
    y_pred = dba_km.fit_predict(X_train)

    for yi in range(num_cluster):
        ratio_each_cluster = 1.0*y_pred[y_pred==yi].shape[0]/ y_length
        array_dba.append(ratio_each_cluster)
        plt.subplot(3, num_cluster, num_cluster + 1 + yi)
        for xx in X_train[y_pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.2)
        plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-")
        plt.xlim(0, sz)
        plt.ylim(-4, 4)
        plt.text(0.55, 0.85,'Cluster %d' % (yi + 1) + ': %2.1f%%' % (100*ratio_each_cluster),
                 transform=plt.gca().transAxes)
        if yi == 1:
            plt.title("DBA $k$-means")

    dba_ari = adjusted_rand_score(y_train, y_pred)
    dba_nmi = normalized_mutual_info_score(y_train, y_pred)

    # Soft-DTW-k-means
    print("Soft-DTW k-means")
    sdtw_km = TimeSeriesKMeans(n_clusters=num_cluster,
                               metric="softdtw",
                               metric_params={"gamma": .01},
                               verbose=True,
                               random_state=seed)
    y_pred = sdtw_km.fit_predict(X_train)

    for yi in range(num_cluster):
        ratio_each_cluster = 1.0*y_pred[y_pred==yi].shape[0]/ y_length
        array_soft_dba.append(ratio_each_cluster)
        plt.subplot(3, num_cluster, 2*num_cluster + 1 + yi)
        for xx in X_train[y_pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.2)
        plt.plot(sdtw_km.cluster_centers_[yi].ravel(), "r-")
        plt.xlim(0, sz)
        plt.ylim(-4, 4)
        plt.text(0.55, 0.85,'Cluster %d' % (yi + 1) + ': %2.1f%%' % (100*ratio_each_cluster),
                 transform=plt.gca().transAxes)
        if yi == 1:
            plt.title("Soft-DTW $k$-means")
    soft_dtw_ari = adjusted_rand_score(y_train, y_pred)
    soft_dtw_nmi = normalized_mutual_info_score(y_train, y_pred)

    print(num_cluster)
    print(array_kmeans)
    print(array_dba)
    print(array_soft_dba) # = [])

    dict_kmeans[str(num_cluster)] = array_kmeans
    dict_dba[str(num_cluster)] = array_dba
    dict_softdtw[str(num_cluster)] = array_soft_dba

    dict_kmeans_ari[str(num_cluster)] = kmeans_ari
    dict_dba_ari[str(num_cluster)] = dba_ari
    dict_softdtw_ari[str(num_cluster)] = soft_dtw_ari

    dict_kmeans_nmi[str(num_cluster)] = kmeans_nmi
    dict_dba_nmi[str(num_cluster)] = dba_nmi
    dict_softdtw_nmi[str(num_cluster)] = soft_dtw_nmi


    plt.tight_layout()
    if not os.path.exists(fig_folder):
        os.mkdir(fig_folder)

    if savefig:
        plt.savefig(os.path.join(fig_folder,
                                 'example_three_ts_clustering_numClusters'+str(num_cluster)+'.png')
                )
    #plt.show()



d = 1

ari_kmeans = []
ari_dba = []
ari_softdtw = []


for num_cluster in num_clusters_array:
    ari_kmeans.append(dict_kmeans_ari[str(num_cluster)])
    ari_dba.append(dict_dba_ari[str(num_cluster)])
    ari_softdtw.append(dict_softdtw_ari[str(num_cluster)])

plt.figure()
plt.plot(num_clusters_array, ari_kmeans, 'b')
plt.plot(num_clusters_array, ari_dba, 'y')
plt.plot(num_clusters_array, ari_softdtw, 'r')
plt.legend(('k-means','DBA','Soft-DTW'))
plt.xlabel('Number of Clusters')
plt.ylabel('Adjusted Rand Score')
#plt.savefig('ari_score.png')
plt.show()



ari_kmeans = []
ari_dba = []
ari_softdtw = []


for num_cluster in num_clusters_array:
    ari_kmeans.append(dict_kmeans_nmi[str(num_cluster)])
    ari_dba.append(dict_dba_nmi[str(num_cluster)])
    ari_softdtw.append(dict_softdtw_nmi[str(num_cluster)])

plt.figure()
plt.plot(num_clusters_array, ari_kmeans, 'b')
plt.plot(num_clusters_array, ari_dba, 'y')
plt.plot(num_clusters_array, ari_softdtw, 'r')
plt.legend(('k-means','DBA','Soft-DTW'))
plt.xlabel('Number of Clusters')
plt.ylabel('Normalized Mutual Information (NMI) Score')
plt.savefig('NMI_score.png')
plt.show()

max_ratio_kmeans = []
max_ratio_dba = []
max_ratio_soft_dba = []


for num_cluster in num_clusters_array:
    array_kmeans_ratio = dict_kmeans[str(num_cluster)]
    array_dba_ratio = dict_dba[str(num_cluster)]
    array_soft_dba_ratio = dict_softdtw[str(num_cluster)]
    max_ratio_kmeans.append( max(array_kmeans_ratio))
    max_ratio_dba.append( max(array_dba_ratio))
    max_ratio_soft_dba.append(max(array_soft_dba_ratio))

plt.figure()
plt.plot(num_clusters_array, max_ratio_kmeans)
plt.plot(num_clusters_array, max_ratio_dba)
plt.plot(num_clusters_array, max_ratio_soft_dba)
plt.legend(('k-means','DBA','Soft-DTW'))
plt.show()


min_ratio_kmeans = []
min_ratio_dba = []
min_ratio_soft_dba = []


for num_cluster in num_clusters_array:
    array_kmeans_ratio = dict_kmeans[str(num_cluster)]
    array_dba_ratio = dict_dba[str(num_cluster)]
    array_soft_dba_ratio = dict_softdtw[str(num_cluster)]
    min_ratio_kmeans.append( min(array_kmeans_ratio))
    min_ratio_dba.append( min(array_dba_ratio))
    min_ratio_soft_dba.append(min(array_soft_dba_ratio))



median_ratio_kmeans = []
median_ratio_dba = []
median_ratio_soft_dba = []

import numpy as np

for num_cluster in num_clusters_array:
    array_kmeans_ratio = dict_kmeans[str(num_cluster)]
    array_dba_ratio = dict_dba[str(num_cluster)]
    array_soft_dba_ratio = dict_softdtw[str(num_cluster)]
    median_ratio_kmeans.append( np.median(np.array(array_kmeans_ratio)))
    median_ratio_dba.append( np.median(np.array(array_dba_ratio)))
    median_ratio_soft_dba.append(np.median(np.array(array_soft_dba_ratio)))

def ratio_to_percent(list_of_ratios):
    list_of_ratios = [100*ratio for ratio in list_of_ratios]
    return list_of_ratios

def percent_to_ratio(list_of_ratios):
    list_of_ratios = [1.0*ratio/100 for ratio in list_of_ratios]
    return list_of_ratios

max_ratio_kmeans = ratio_to_percent(max_ratio_kmeans)
median_ratio_kmeans = ratio_to_percent(median_ratio_kmeans)
min_ratio_kmeans = ratio_to_percent(min_ratio_kmeans)

max_ratio_dba = ratio_to_percent(max_ratio_dba)
min_ratio_dba = ratio_to_percent(min_ratio_dba)
median_ratio_dba = ratio_to_percent(median_ratio_dba)

max_ratio_soft_dba = ratio_to_percent(max_ratio_soft_dba)
min_ratio_soft_dba = ratio_to_percent(min_ratio_soft_dba)
median_ratio_soft_dba = ratio_to_percent(median_ratio_soft_dba)

"""

max_ratio_soft_dba = percent_to_ratio(max_ratio_soft_dba)
min_ratio_soft_dba = percent_to_ratio(min_ratio_soft_dba)
median_ratio_soft_dba = percent_to_ratio(median_ratio_soft_dba)
"""

fig, ax = plt.subplots(1,1)
ax.fill_between(num_clusters_array, max_ratio_kmeans, min_ratio_kmeans, alpha = 0.1, color = 'b')
ax.plot(num_clusters_array, median_ratio_kmeans, color = 'b')
ax.fill_between(num_clusters_array, max_ratio_soft_dba, min_ratio_soft_dba, alpha = 0.1, color = 'r')

ax.plot(num_clusters_array, median_ratio_soft_dba, color = 'r')

ax.fill_between(num_clusters_array, max_ratio_dba, min_ratio_dba, alpha = 0.1, color = 'y')
ax.plot(num_clusters_array, median_ratio_dba, color = 'y')
ax.legend(('k-means (Median)', 'Soft-DTW (Median)', 'DBA (Median)',
           'k-means','Soft-DTW','DBA'), bbox_to_anchor=(1.3,1), loc="upper right")
plt.xlabel('Number of Clusters')
plt.ylabel('Percentage of each cluster')
plt.tight_layout()
plt.savefig('diff_methods_cluster_sizes.png')
plt.show()

"""

plt.scatter(num_clusters_array, max_ratio_kmeans, edgecolors= 'b', marker= 'x')
plt.scatter(num_clusters_array, median_ratio_kmeans, edgecolors= 'b', marker= 'o')
plt.scatter(num_clusters_array, min_ratio_kmeans, edgecolors= 'b', marker= 's')


plt.scatter(num_clusters_array, max_ratio_dba , edgecolors= 'r', marker= 'x')
plt.scatter(num_clusters_array, median_ratio_dba, edgecolors= 'r', marker= 'o')
plt.scatter(num_clusters_array, min_ratio_dba, edgecolors= 'r', marker= 's')

plt.scatter(num_clusters_array, max_ratio_soft_dba, edgecolors= 'k', marker= 'x')
plt.scatter(num_clusters_array, median_ratio_soft_dba, edgecolors= 'k', marker= 'o')
plt.scatter(num_clusters_array, min_ratio_soft_dba, edgecolors= 'k', marker= 's')

#plt.legend(('k-means','DBA','Soft-DTW'))
plt.show()
"""

# measure performances of clustering

#from sklearn.metrics.cluster import adjusted_rand_score
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html

# 0 - randomly labeling
# 1 - exact identical and allows random permutation
#adjusted_rand_score(labels_true, labels_pred)