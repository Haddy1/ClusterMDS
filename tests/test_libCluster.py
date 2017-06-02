import sys
sys.path.append("../")
import lib.libCluster as libC
from lib.libDistanceMatrix import DistanceMatrix
import random
import numpy as np
import scipy

###  random relabel ###
##check if nr entries remains the same
def test_calcDistMatrix():
    cube = np.array([[0,0,0], [0,0,1], [0,1,0], [0,1,1],[1,0,0],[1,0,1], [1,1,0], [1,1,1]])
    cdist = scipy.spatial.distance.cdist(cube,cube)
    label_indices = ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b']
    clusters = libC.labelPoints(cube, label_indices)
    dist_matrix = libC.calcDistMatrix(clusters)
    if np.array_equal(cdist, dist_matrix):
        return True
    else:
        return False

def test_getIndexList():
    label_indices = ['a', 'b', 'a', 'a', 'b', 'a', 'b', 'b']
    labels = ['a', 'b']
    index_list = libC.getIndexList(labels, label_indices)
    if index_list['a'] == [0,2,3,5] and index_list['b'] == [1,4,6,7]:
        return True
    else: return False

def test_RandomRelabel():
    #create test list
    nr_entries=1000
    index_list = {}
    for i in range(0, 10):
        index_list[str(i)] = []
    for i in range(0,nr_entries):
        label = str(random.randint(0,9))
        index_list[label].append(random.randint(0,1000))

    new_list = libC.randomReLabel(index_list.copy())
    nr_entries_relabeled=0
    for label, indices in new_list.items():
        nr_local_entries=0
        nr_local_entries_relabeled=0
        nr_local_entries = len(index_list[label])
        nr_local_entries_relabeled = len(indices)
        nr_entries_relabeled += nr_local_entries_relabeled

        assert nr_local_entries_relabeled == nr_local_entries, "randomRelabel test failed - local entries do not match for label " + label
    assert nr_entries_relabeled == nr_entries, "random relabel test failed"
    return True

def test_calcMeanDist():
    cube = np.array([[0,0,0], [0,0,1], [0,1,0], [0,1,1],[1,0,0],[1,0,1], [1,1,0], [1,1,1]])
    dist = np.dot(cube, cube.T)
    index_list = {'a': [0,1,2], 'b': [3,4,5]}
    mean = libC.calcMeanDist(index_list, dist)
    means = np.zeros((2,2))
    means[0][0] = dist[0:3, 0:3].mean()
    means[0][1] = dist[0:3, 3:6].mean()
    means[1][0] = dist[3:6, 0:3].mean()
    means[1][1] = dist[3:6, 3:6].mean()
    if np.array_equal(mean, means): return True
    else : return False

def test_discriminationValue():
    data = DistanceMatrix(['a','b'], np.array([[1.0,2.0],[2.0,4.0]]))
    delta_0 = data[0][0] + data[1][1] - data[1][0] - data[0][1]
    delta = libC.discriminationValue(['a','b'], data)
    if delta == delta_0: return True
    else:
        print("delta_0: " + str (delta_0))
        print("delta: " + str (delta))

        return False







print ("test_calcDistMatrix: " + str(test_calcDistMatrix()))
print ("test_getIndexList: " + str(test_getIndexList()))
print ("test_RandomRelabel: " + str(test_RandomRelabel()))
print ("test_calcMeantMatrix: " + str(test_calcDistMatrix()))
print ("test_discriminationValue: " + str(test_discriminationValue()))
