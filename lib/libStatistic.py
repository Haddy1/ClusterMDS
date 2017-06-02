import os
import PyQt5
from PyQt5.QtCore import QObject
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
import scipy
from .libCluster import *
from .libDistanceMatrix import DistanceMatrix
from matplotlib.figure import Figure

class Statistic(QFrame):
    """ Widget for calculating and showing statistic information

    Calculates and Displays the p-value estimates for the separation quality of the selected clusters

    Parameters
    ---------
    labels: list
        contains the labels, for which the p-values should be calculated

    distance_matrix: DistanceMatrix
        Contains the euclidian distances between all datapoints

    plotWindow: QWidget
        Window to which results should be rendered. Needs a valid layout

    options: Options
        Option Class containing  nr of desired permutations
    """
    clusters = None
    mean_dist_matrix = None
    p_values = None
    p_total = None
    def __init__(self,labels, distance_matrix, plotWindow, options):
        super(Statistic, self).__init__()
        self.distance_matrix = distance_matrix
        self.labels = labels
        self.plotWindow = plotWindow
        self.permutations = options.permutations
        self.options = options
        self.initUi()

        self.startCalculation()

    def startCalculation(self):
        """ Starts Statistic Calculation in StatisticThread """

        #start progress bar
        self.progressBar = QProgressBar(self)
        self.progressBar.setValue(0)
        nr_p_values = self.combinations(len(self.labels)) +1
        self.layout.addWidget(self.progressBar)
        self.progressBar.setRange(0,nr_p_values)

        #spawn new thread and register return signal
        self.calc_thread = StatisticThread(self.labels, self.distance_matrix, self.permutations)
        self.calc_thread.finished.connect(self.callbackFunction)
        self.calc_thread.progress.connect(self.callback_progress)
        self.calc_thread.start()

    def callbackFunction(self, p_total, p_values, mean_dist_matrix):
        """ Callback Function for finished background WorkerThread

        calls showValues

        Parameters
        ---------
        p_total: double
            p-value for all clusters
        p_values: list
            p-value for each pair of Clusters
        mean_dist_matrix: DistanceMatrix
            Contains the mean for each Cluster Distance combination
        """
        #remove progress bar
        if self.progressBar:
            self.layout.removeWidget(self.progressBar)
            self.progressBar.deleteLater()
            self.progressBar = None
        self.p_total = p_total
        self.p_values = p_values
        self.mean_dist_matrix = mean_dist_matrix
        self.showValues()
        self.saveButton.setEnabled(True)
        #fig1 = Figure()
        #ax1f1 = fig1.add_subplot(111)
        #self.plotWindow.addmpl(fig1)
        ##print (deltas)
        #ax1f1.plot(self.deltas, '.')

    def callback_progress(self, progression):
        """ Updates Progress Bar """

        self.progressBar.setValue(progression)


    def initUi(self):
        """ Initialize Elements """

        self.setFrameStyle(QFrame.StyledPanel| QFrame.Sunken)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.toolbar = QToolBar()
        self.layout.addWidget(self.toolbar)
        closeButton = QAction('Close', self)
        closeButton.triggered.connect(self.close)
        self.saveButton = QAction('Save', self, enabled = False)
        self.saveButton.triggered.connect(self.saveStats)

        self.toolbar.addAction(closeButton)
        self.toolbar.addAction(self.saveButton)

    def showValues(self):
        """
        displays calculated p-values in a table

        """

        self.p_widget = QLabel()
        self.layout.addWidget(self.p_widget)

        self.p_widget.setText(str("total p_value: \n" + str(self.p_total)))

        nr_combinations = self.combinations(len(self.labels))
        table = QTableWidget(nr_combinations, 1)
        table.setHorizontalHeaderLabels(["p_value"])

        #make table read-only
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)


        rowLabels = []
        row = 0
        for indexA, labelA in enumerate(self.labels):
            for indexB, labelB in enumerate(self.labels):
                if indexA < indexB:
                    rowLabels.append(str(labelA[0:10] + '\n' + labelB[0:10]))
                    p_item = QTableWidgetItem(str(self.p_values[labelA][labelB]))
                    table.setItem(row, 0,  p_item)
                    row += 1
        table.setVerticalHeaderLabels(rowLabels)
        self.layout.addWidget(table)
        self.accuracy = QLabel()
        self.layout.addWidget(self.accuracy)
        self.accuracy.setText("accuracy: 1 - " + '{:.2e}'.format(1.0/self.permutations))




    def saveStats(self):
        """
        writes p-values in file
        """

        assert self.p_total is not None, "p_total does not exist"
        assert self.p_values is not None, "no p_values exist"

        #get filename
        filename, _ = QFileDialog.getSaveFileName(self, 'Save File', os.getenv('HOME'))

        basename, extension = os.path.splitext(filename)


        filename_p_values = str(basename) + "_pvalues" + extension
        #write total p_value
        with open(filename_p_values, 'w') as f:
            f.write(str(self.p_total) + "\n")

        #open as binary so numpy appends to file
        with open(filename_p_values, 'ab') as f:
            np.savetxt(f, self.p_values)
        filename_distances = str(basename) + "_mean_distances" + extension

        with open(filename_distances, 'wb') as f:
            np.savetxt(f, self.mean_dist_matrix)

    def combinations(self, x):
        """
        returns nr of possible combinations for x entries
        """
        y = 0
        for i in range(0, x):
            y += i
        return y


#Worker thread for calculation, so that the ui doesnt freeze and we can show a progress bar
class StatisticThread(QtCore.QThread):
    """
    Worker Thread for background statistics Calculation

    Provides a p-value estimate for the seperation quality of the selected clusters, by randomly rassigning the points to the clusters several times

    Parameters
    ----------
    labels: list
        labels of the clusters for which the seperation quality p-value estimation is calculated
    distance_matrix: array-like
        Distance matrix for all clusters
    permutations: float, optional, default: 10000
        Nr of times random reassigning should occur, determines the accuracy of p-values

    Results
    ------
    p_total: float
        seperation quality for all selected clusters
    p_values: list
        p_values for each combination of the selected clusters
    mean_dist_matrix: DistanceMatrix
        Contains the mean for each Cluster Distance combination

    Signals
    -------
    finished(p_total, p_values, mean_dist_matrix)
        sends the results to Statistic.callbackFunction

    progress(progression)
        updates the Progress bar for each finished p-value
    """

    finished = QtCore.pyqtSignal('PyQt_PyObject', 'PyQt_PyObject', 'PyQt_PyObject')
    progress = QtCore.pyqtSignal('PyQt_PyObject')

    #initialize thread with the high dimensional data
    def __init__(self, labels, distance_matrix, permutations=10000):
        QtCore.QThread.__init__(self)
        self.labels = labels
        self.distance_matrix = distance_matrix
        self.progression = 0
        self.permutations = permutations

    def run(self):
        """
        runs calculation
        called automatically as soon as the thread is started"""

        p_total, mean_dist_matrix = self.calcDiscriminations(self.labels, self.distance_matrix, self.permutations)
        p_values = self.clusterStatistics(self.labels)


        #send calculated data back to main thread
        self.finished.emit(p_total, p_values, mean_dist_matrix)

        #exit thread
        self.exit()

    def clusterStatistics(self, labels):
        """ calculates p_value estimate for seperation quality for each combination of clusters

        Parameters
        ----------
        labels: list
            Labels for which p_values should be calculated

        Returns
        -------
        p_values: DistanceMatrix
            Matrix containing the p_values for each cluster combination
        """

        #create new DistanceMatrix as container
        p_values = DistanceMatrix(labels)

        for indexA, labelA in enumerate(labels):
            for  indexB, labelB in enumerate(labels):
                if indexA < indexB:
                    p_values[labelA][labelB], _ = self.calcDiscriminations([labelA, labelB], self.distance_matrix, self.permutations)

                    p_values[labelB][labelA] = p_values[labelA][labelB]

        return p_values

    def calcDiscriminations(self, labels, distance_matrix, permutations=10000):
        """
        calculates p-value estimate for the seperation quality of labeled clusters

        calculates the original discriminationValue delta_0
        then randomly reassignes the points to the clusters
        each time recalculating the discriminationValue delta_k
        The number of times delta_k is below the original value delta_0 is counted
        and divided by total nr of recalculations

        This provides a p-value estimate of how good the clusters are seperated.

        Parameters
        ----------
        labels: list
            labels for the clusters for which the seperation quality should be estimated
        distance_matrix: DistanceMatrix
            labeled distance Matrix containing the distances between all points
        permutations: int, optional, default:10000
            number of times the points should be randomly relabeled
            determines the accuracy of the calculated p-value

        Returns
        -------
        p-value: float
            the calculated p-value indicating the seperation quality between the Clusters
        mean_dist_matrix: DistanceMatrix
            Contains the mean for each Cluster Distance combination


            """

        assert len(labels) > 0, "No labels were selected"


        index_list = getIndexList(labels, distance_matrix.labels)
        mean_dist_matrix = calcMeanDist(index_list, distance_matrix)

        disc_value = discriminationValue(labels, mean_dist_matrix)
        #only one cluster simply return delta
        if len(labels) == 1:
            return  (0, disc_value)

        deltas = np.zeros(permutations)

        index_list_old = index_list
        smaller_values = []
        for i in np.arange(0, permutations):
            rnd_index_list = randomReLabel(index_list)
            rnd_mean_dist_matrix = calcMeanDist(rnd_index_list, distance_matrix)
            deltas[i] = discriminationValue(labels, rnd_mean_dist_matrix)
            if deltas[i] < disc_value:
                smaller_values.append(deltas[i])


        p_value = len(smaller_values) / (permutations * 1.0)

        self.progression +=1
        self.progress.emit(self.progression)

        return (p_value, mean_dist_matrix)
