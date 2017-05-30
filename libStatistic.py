import os
import PyQt5
from PyQt5.QtCore import QObject
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
import scipy
from libCluster import *
from libDistanceMatrix import DistanceMatrix
from matplotlib.figure import Figure

class Statistic(QFrame):
    """ Widget for calculating and showing statistic information

    Upon creating it
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
        self.progressBar.setValue(progression)


    def initUi(self):
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
        y = 0
        for i in range(0, x):
            y += i
        return y


#Worker thread for calculation, so that the ui doesnt freeze and we can show a progress bar
class StatisticThread(QtCore.QThread):
    finished = QtCore.pyqtSignal('PyQt_PyObject', 'PyQt_PyObject', 'PyQt_PyObject')
    progress = QtCore.pyqtSignal('PyQt_PyObject')

    #initialize thread with the high dimensional data
    def __init__(self, labels, distance_matrix, permutations=10000):
        QtCore.QThread.__init__(self)
        self.labels = labels
        self.distance_matrix = distance_matrix
        self.progression = 0
        self.permutations = permutations

    #this function is called automatically as soon as the thread is started
    def run(self):
        p_total, mean_dist_matrix = self.calcDiscriminations(self.labels, self.distance_matrix, self.permutations)
        p_values = self.clusterStatistics(self.labels)[0]

        #send calculated data back to main thread
        self.finished.emit(p_total, p_values, mean_dist_matrix)

        #exit thread
        self.exit()

    def clusterStatistics(self, labels):

        #create new DistanceMatrix as container
        p_values = DistanceMatrix(labels)

        for indexA, labelA in enumerate(labels):
            for  indexB, labelB in enumerate(labels):
                if indexA < indexB:
                    p_values[labelA][labelB], deltas = self.calcDiscriminations([labelA, labelB], self.distance_matrix, self.permutations)

                    p_values[labelB][labelA] = p_values[labelA][labelB]

        return p_values, deltas

    def calcDiscriminations(self, labels, distance_matrix, permutations=10000):
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


def permute(self, x, condensed=False):
        order = np.random.permutation(x.shape[0])
        permuted = x[order][:, order]
        if condensed:
            permuted = scipy.spatial.distance.squareform(permuted)
        return permuted
