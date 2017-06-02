import sys
import numpy as np
import scipy
from sklearn.cluster import KMeans
import collections

from libDistanceMatrix import DistanceMatrix


def labelPoints(points, label_indices):
    """"
    Creates a new datastructure containing both labels and points using python dictionary

    Parameters
    ----------
    points: ndarray
        contais the data points
    labels: list
        list of labels, each entry corresponding to a point

    Returns
    --------
    clusters: dictionary
        with the labels as keys and the respective points as values

    """

    clusters = collections.OrderedDict() #contains the labeled points (clusters)
    for index in np.arange(len(label_indices)):
        label = str(label_indices[index])
            #look for cluster with this label
        if label in clusters:
            #add new point to selected cluster
            clusters[label] = np.append(clusters.get(label),np.array([points[index]]), axis=0)
        else:
            #create new cluster
            clusters[label] = np.array([points[index]])


    #check if all points were labeled
    nr_clusters = 0
    for label in clusters:
        nr_clusters +=  clusters[label].shape[0]
#    assert nr_clusters == points.shape[0], "Not all points were labeled"


    return clusters

def calcDistMatrix(clusters):
    """Calculates Distance Matrix

    Parameters
    ----------
    clusters: dictionary containing labeled points


    Returns
    -------
    dist_matrix: DistanceMatrix
    labeled distances between all points in clusters
    """

    label_list = []
    ordered_matrix = None
    for label, cluster in clusters.items():
        if ordered_matrix is None:
            ordered_matrix = cluster
        else:
            ordered_matrix = np.append(ordered_matrix, cluster, axis=0)
        for _ in range(cluster.shape[0]):
            label_list.append(label)
    dist_matrix = DistanceMatrix(label_list, scipy.spatial.distance.cdist(ordered_matrix, ordered_matrix))

    return dist_matrix




def discriminationValue(labels, mean_dist_matrix):
    """"
    calculates the discrimination Value between the clusters from the mean distance matrix

    Parameters
    ----------
    labels: list
        list containing the labels of the clusters
    mean_dist_matrix: DistanceMatrix
        matrix with the means of the intra and inter distances of the clusters

    Returns
    --------
    delta: float
        discriminationValue for the clusters

    """


    delta = 0

    assert len(labels) > 0, "labels are empty"
    if len(labels) == 1:
        return mean_dist_matrix[0][0]

    weigth = 2.0/(len(labels)-1)

    size = mean_dist_matrix.shape[0]

    delta = np.diag(mean_dist_matrix).sum()
    delta -= weigth * mean_dist_matrix[np.triu_indices(mean_dist_matrix.shape[0], 1)].sum()

    return delta


def getIndexList(labels, label_indices):
    """
    for each label get position of points in distance matrix

    Parameters
    ----------
    labels: list
        desired labels
    label_indices: list
        list of labels contained in distance matrix


    Returns
    ------
    index_list: dictionary
    contains the index positions in distance matrix for each label

    """

    index_list = {}
    for label in labels:
        index_list[label] = []
    for index, label in enumerate(label_indices):
        if label in  labels:
            index_list[label].append(index)
    return index_list

def randomReLabel (index_list):
    """
    permutates the entries in index_list
    this is equivalent to reassigning the points to the clusters

    Parameters
    ---------
    index_list: list
        original point indices

    Returns
    -------
    new_indices: list
        relabeled point indices

    """
    combined_list = []
    for label, indices in index_list.items():
        combined_list.extend(indices)

    new_combined = np.random.permutation(combined_list)

    new_indices = {}
    start = 0
    for label, indices in index_list.items():
        new_indices[label] = list(new_combined[start:(start + len(indices))])
        start += len(indices)


    return new_indices



def calcMeanDist(index_list, dist_matrix):
    """
    Calculates the Mean of Intra and Inter Cluster distances

    Parameters
    ----------
    index_list: dictionary
        for each label it contains the index of the points in dist_matrix
    dist_matrix: DistanceMatrix
        Distance Matrix of high Dimensional data

    Returns
    -------
    mean_dist_matrix: DistanceMatrix
        labeled DistanceMatrix,
        each entry is the Mean of the respective Cluster distance
    """



    labels = list(index_list.keys())

    mean_dist_matrix = DistanceMatrix(labels)
    for  labelA, indicesA in index_list.items():
        for labelB, indicesB in index_list.items():

            dist_matrix_rows = dist_matrix[indicesA]
            mean = dist_matrix_rows[:,indicesB].mean()
            mean_dist_matrix[labelA,labelB] = dist_matrix_rows[:,indicesB].mean()


    return mean_dist_matrix


#Uses KMeans alghorithm to find the clusters.
#data:  n-dimensional data points as numpy array
#n_clusters: the number of clusters the alghorithm has to find
#returns: list containing the labels corresponding to each point in data
def automaticLabeling(data , n_clusters):
    kmeans = KMeans(n_clusters).fit(data)
    return kmeans.labels_.tolist()
