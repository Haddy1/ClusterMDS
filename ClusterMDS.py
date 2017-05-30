#!/usr/bin/python3
import numpy as np
import sys
import os

from PyQt5.QtCore import QObject
from PyQt5.uic import loadUiType
from PyQt5.QtGui import QIcon
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
import collections
from collections import OrderedDict

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar)


Ui_MainWindow, QMainWindow = loadUiType('window.ui')
Ui_Options, QOptions= loadUiType('options.ui')

import libMDS
import libCluster
import libStatistic
from libDistanceMatrix import DistanceMatrix
from scipy.spatial.distance import cdist


#This File contains all the UI stuff.

class Main(QMainWindow, Ui_MainWindow):
    markersize = 6.5 # standard marker size
    annotated_point = None #current shown annotation for point [xy, annotation]
    selected_labels = [] #labels, for the currently selected clusters
    mds_data = np.empty(0) #array with md scaled data
    unrotated_mds_data = None #when rotation is selected this save original data
    distance_matrix = None
    data = None #original data array
    label_indices = None #labels list for the clusters
    orig_label_indices = None
    clusters = None #dictionary containing the mds_data for each label
    canvas = [] #list containing the plots
    progressBar = None
    calc_thread = None
    time_offset = 0


    def __init__(self, ):
        super(Main, self).__init__()
        self.setupUi(self)
        self.options = Options()
        self.options.doRotation.connect(self.doRotation)
        self.initUI()

    def initUI(self):
        """ Sets up menu and Buttons
        """
        openButton = QAction('Open', self)
        openButton.setShortcut('Ctrl+O')
        openButton.setStatusTip('Load new dataset')
        openButton.triggered.connect(self.openFile)
        saveButton = QAction('Save', self)
        saveButton.setShortcut('Ctrl+S')
        saveButton.setStatusTip('Save scaled dataset')
        saveButton.triggered.connect(self.saveFile)
        optionsButton = QAction('Options', self)
        optionsButton.setStatusTip('Additional Options')
        optionsButton.triggered.connect(self.showOptions)

        self.statisticButton = QAction('Statistics', self, enabled=False)
        self.statisticButton.setStatusTip('show statistics')
        self.statisticButton.triggered.connect(self.showStatistics)
        self.labelButton= QAction('Load Label', self, enabled=False)
        self.labelButton.setStatusTip('load label data')
        self.labelButton.triggered.connect(self.loadLabels)
        self.onoffButton= QAction('OnOff Times', self, enabled=False)
        self.onoffButton.setStatusTip('discard timestamps')
        self.onoffButton.triggered.connect(self.remove_timestamps)
        self.deleteButton =  QAction('Delete Cluster', self, enabled=False)
        self.deleteButton.triggered.connect(self.deleteCluster)

        self.toolbar.addAction(openButton)
        self.toolbar.addAction(saveButton)
        self.toolbar.addAction(optionsButton)
        self.toolbar.addAction(self.statisticButton)
        self.toolbar.addAction(self.labelButton)
        self.toolbar.addAction(self.onoffButton)
        self.toolbar.addAction(self.deleteButton)

    #opens a Message Box with an error message
    def errDialog(self, err_msg):
        """
        opens a Message Box and displays an error message
        """

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText(err_msg)
        msg.setWindowTitle("Error")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()


    def openFile(self):
        """
        Shows a File Selector Dialog and reads in the data from the selected file
        Then it starts the mds calcluation by calling calcMDS
        """

        self.time_offset = 0


        #Apparently QThreads must finish themselves and cannot be killed from the main thread. Since we use an external function we cannot do any handling for that. So just display this error massage. (Could be fixed by using the multithreading module instead, but I could not be asked to implement this)
        if self.calc_thread:
            if (self.calc_thread.isRunning()):
                self.errDialog("Sorry calculation is still running. \nPlease wait for it to finish or restart the application.")
                return

        filenames = QFileDialog.getOpenFileNames(self, 'Open Files', os.getenv('HOME'))



        data = np.empty((0,0))
        labels = []
        nr_labels = 0

        #read in all points and create labels for each file
        for filename in filenames[0]:
            #open as numpy array
            try:
                cluster = np.loadtxt(filename)
            except IOError:
                continue
            if nr_labels == 0:
                data = cluster
            else:
                #merge data points into the data array
                data = np.append(data ,cluster, axis=0)

            #add a label for each data point
            for point in np.arange(cluster.shape[0]):
                labels.append(os.path.basename(filename))
            nr_labels += 1

        #no labels -> no correct data
        if nr_labels == 0:
            self.errDialog("No valid data could be found")
            return

        #remember the data
        self.data = data
        self.label_indices = labels


        print (data.shape)

        #remember the file name, so that we can later use it as title for the plot
        #only one file
        if nr_labels == 1:
            #remember the filename
            self.data_name = labels[0]

        #multiple files
        else:
            #remember the dirname
            self.data_name = os.path.basename(os.path.dirname(filenames[0][0]))

        #remove previous plot
        for plot in self.canvas:
            self.rmmpl()

        #start calculating
        self.mds_data = self.calcMDS(self.data)



    #saves the current mds transformed data
    def saveFile(self):
        """
        Saves the MDS transformed data matrix into a selected directory, one file for each cluster
        """
        dirname = QFileDialog.getExistingDirectory(self, 'Save File', os.getenv('HOME'))

        assert self.clusters and self.mds_data, "Trying to save empty data"

        #first save all points in single file
        np.savetxt(dirname + '/' + MDS_all-clusters.txt, self.mds_data)
        #then create a file for each cluster
        for label, cluster in self.clusters.items():
            try:
                np.savetxt(str(dirname + '/' + label + "_MDS.txt"), cluster)
            except IOError:
                self.errDialog(str(label + "Could not be saved"))

    #loads a file containing the labels corresponding to the data points
    def loadLabels(self):
        """
        loads a file containing the labels corresponding to the data points
        """
        if not self.clusters:
            self.errDialog("Please open datafile first")

        filename, _ = QFileDialog.getOpenFileName(self, 'Open File', os.getenv('HOME'))
        try:
            with open(filename) as file:
                labels = list(filter(None, (line.rstrip() for line in file)))
        except IOError:
             self.errDialog(str("Could not Open" + filename))
             return


        if len(labels) != self.mds_data.shape[0]:
            self.errDialog("number of labels dies not equal the number of points")
            return

        self.label_indices = labels
        #relabel points
        self.clusters = libCluster.labelPoints(self.mds_data, self.label_indices)

        #replot
        self.rmmpl()
        self.plotLabel(self.clusters)

    #spawns the thread used for mds calculation
    def calcMDS(self, data):

        if data.shape[0] < 1:
            self.errDialog("Input Data does not contain any points")
            return
        if data.shape[1] < 2:
            self.errDialog("Input Data must have at least 2 Dimensions")
            return

        #start progress bar
        self.progressBar = QProgressBar(self)
        self.progressBar.setRange(0,1)
        self.mplvl.addWidget(self.progressBar)
        self.progressBar.setRange(0,0)

        start = self.options.offset_start
        end= self.options.offset_end

        #spawn new thread and register return signal
        self.calc_thread = WorkerThread(data[start:-end])
        self.calc_thread.finished.connect(self.callbackFunction)
        self.calc_thread.exception.connect(self.mdsException)
        self.calc_thread.start()


    #creates the plot showing the scatter image of the mds transformed data
    def plotLabel(self, clusters):
        fig1 = Figure()
        ax1f1 = fig1.add_subplot(111)
        self.ax1f1 = ax1f1
        main.addmpl(fig1)

        self.statisticButton.setEnabled(True)
        self.onoffButton.setEnabled(True)
        self.labelButton.setEnabled(True)




        #plot the clusters
        legend = []
        lines = []
        for label, cluster in clusters.items():
            line = ax1f1.plot(cluster[:,0], cluster[:,1], "o", picker=5, alpha=0.5, markersize=self.markersize, gid = label)

            #remember the line object for the cluster
            lines.append(line)
            #add label text for the legend
            legend.append(label[0:10])


        #add legend
        mleg = ax1f1.legend(legend)
        #add file/dirname as title
        ax1f1.set_title(self.data_name)

        #connect eventhandling to plot
        fig1.canvas.mpl_connect('pick_event', self.onPick)
        fig1.canvas.mpl_connect('motion_notify_event', self.onHover)


        #create a dictionary, so that we can later indentify the label from the line object
        self.legend_dict = {}
        self.line_dict = {}
        for legend_line, line, label in zip(mleg.get_lines(), ax1f1.get_lines(), clusters):

            #set identifier for each line object
            gid_line = str(label)
            line.set_gid(gid_line)
            self.legend_dict[gid_line] = {'legend': legend_line, 'line': line, 'label': label}



    #on click on data point, highlight all points in the same cluster and insert the label in selected_labels list
    def onPick(self, event):
        thisline = event.artist

        line_info = self.legend_dict[str(thisline.get_gid())]

        marker = thisline.get_markersize()
        #point was not selected before
        if marker == self.markersize:
            #draw bigger marker with edge
            thisline.set_markeredgecolor("k")
            thisline.set_markeredgewidth(2)
            thisline.set_markersize(10)
            #also redraw corresponding legend entry
            line_info['legend'].set_markeredgecolor("k")
            line_info['legend'].set_markeredgewidth(2)
            line_info['legend'].set_markersize(10)

            #add label to selected labels
            self.selected_labels.append(line_info['label'])

            #enable buttons
            self.statisticButton.setEnabled(True)
            self.onoffButton.setEnabled(True)

            self.deleteButton.setEnabled(True)
        else:
            #reset marker size and color
            thisline.set_markeredgewidth(0.2)
            thisline.set_markersize(self.markersize)
            line_info['legend'].set_markeredgewidth(0.5)
            line_info['legend'].set_markersize(self.markersize)
            #if this point was previously selected remove its label from list
            if self.selected_labels.__contains__(line_info['label']):
                self.selected_labels.remove(line_info['label'])
                #if no label remains in list, reset buttons
                if len(self.selected_labels) == 0:
                    self.deleteButton.setEnabled(False)

        #update drawing
        self.canvas[0].draw_idle()

    #for the point under the mousecursor show its position in the cluster
    def onHover(self, event):
        #find the line object we are hovering over
        for line in self.ax1f1.get_lines():
            if line.contains(event)[0]:
                #get position
                index = line.contains(event)[1]['ind'][0]
                #save coordinates of the point
                xy = line.get_xdata()[index], line.get_ydata()[index]

                #no annotation exists or a new point is getting annotated
                if  not self.annotated_point or self.annotated_point[0] != xy:
                    #delete old annotation
                    if self.annotated_point:
                        self.ax1f1.texts.remove(self.annotated_point[1])
                        self.annotated_point[1].remove
                        self.annotated_point = None

                    #create new annotation containing the index of the point
                    annotation = self.ax1f1.annotate(str(index + self.time_offset), xy=xy, xycoords='data', ha="left", va="bottom", bbox= dict(boxstyle='square,pad=0.3', fc="orange", ec="k", alpha=0.5))
                    self.annotated_point = (xy, annotation)

                    #redraw plot
                    self.canvas[0].draw_idle()

                #delete annotation - probably never gets called
                elif self.annotated_point[0]  != xy:
                    self.annotated_point[1].remove
                    self.annotated_point = None
                    self.canvas[0].draw_idle()


    def addmpl(self, fig):
        """
        Adds a new pyplot Figure to the UI
        """
        self.canvas.append(FigureCanvas(fig))
        self.mplvl.addWidget(self.canvas[-1])
        self.canvas[-1].draw()

        self.toolbar = NavigationToolbar(self.canvas[-1], self.mplwindow, coordinates=True)
        self.mplvl.addWidget(self.toolbar)


    def rmmpl(self):
        """
        deletes latest plot
        """
        #delete plot
        if self.canvas[-1]:
            self.mplvl.removeWidget(self.canvas[-1])
            self.canvas[-1].close()
            self.mplvl.removeWidget(self.toolbar)
            self.toolbar.close()
            del(self.canvas[-1])

        #main plot just got deleted, reset buttons and selections
        if len(self.canvas) == 0:
            self.selected_labels = []
            self.annotated_point = None
            self.statisticButton.setEnabled(False)
            self.onoffButton.setEnabled(False)
            self.deleteButton.setEnabled(False)
            self.labelButton.setEnabled(False)


        #delete progress bar
        if self.progressBar:
            self.mplvl.removeWidget(progressBar)
            self.progressBar.deleteLater()
            self.progressBar = None



    def callbackFunction(self, mds_data):
        """
        The worker thread calls this function, when it finishes the calculation
        creates the cluster container and calls plotLabel
        """

        #remove progress bar
        if self.progressBar:
            self.mplvl.removeWidget(self.progressBar)
            self.progressBar.deleteLater()
            self.progressBar = None

        if self.options.rotate:
            self.unrotated_mds_data = mds_data
            mds_data = libMDS.pca(mds_data, 2)


        start = self.options.offset_start
        end = self.options.offset_end

        self.mds_data = mds_data
        clusters_orig = libCluster.labelPoints(self.data[start:-end], self.label_indices[start:-end])
        self.distance_matrix= libCluster.calcDistMatrix(clusters_orig)

        if self.label_indices:
            self.clusters = libCluster.labelPoints(mds_data, self.label_indices[start:-end])
            self.plotLabel(self.clusters)
        else:
            label_indices = libCluster.automaticLabeling(mds_data, 5)
            self.clusters = libCluster.labelPoints(mds_data, label_indices)
            self.plotLabel(self.clusters)
        self.labelButton.setEnabled(True)
        self.statisticButton.setEnabled(True)
        self.onoffButton.setEnabled(True)

    def deleteCluster(self):
        """
        deletes the selected clusters contained in self.selected_labels
        from the data
        and recalculates the MDS
        """

        #show confirmation box
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText("Really delete label: " + str(self.selected_labels))
        msg.setWindowTitle("deletion")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        btnPressed = msg.exec_()

        if btnPressed == QMessageBox.Ok:
            #delete statistic plot
            if len(self.canvas) > 1:
                self.rmmpl()
                self.statisticButton.setEnabled(False)

            #label our old data points
            orig_clusters = libCluster.labelPoints(self.data, self.label_indices)
            new_labels = []
            nr_labels = 0
            new_data = np.empty(0)
            #create new data array and label list
            for label, cluster in orig_clusters.items():
                if self.selected_labels.__contains__(label):
                    continue
                if nr_labels  == 0:
                    new_data = cluster
                else:
                    #merge data points into the data array
                    new_data = np.append(new_data ,cluster, axis=0)

                #add a label for each data point
                for point in np.arange(cluster.shape[0]):
                    new_labels.append(label)
                nr_labels +=1

            #delete plot
            self.rmmpl()

            #if some data is left, set variables to our new data
            if new_data.any():
                self.mds_data = np.empty(0)
                self.data = new_data
                self.label_indices = new_labels
                #recalculate mds
                self.calcMDS(self.data)

    #shows a plot with the discrimination values
    def showStatistics(self):


        assert self.clusters, "statistics functionality needs already labeled clusters"

        if len(self.selected_labels) == len(self.clusters) or len(self.selected_labels) == 0:
                statistics = libStatistic.Statistic(list(self.clusters.keys()), self.distance_matrix, self, self.options)
                self.splitter.addWidget(statistics)
        else:
            if set(self.selected_labels).issubset(set(self.clusters)):
                statistics = libStatistic.Statistic( self.selected_labels, self.distance_matrix, self, self.options)
                self.splitter.addWidget(statistics)


            else:
                self.errDialog("data does not contain selected labels")



    #callback function for rotation button
    def doRotation(self, rotate):
        #makes only sense if there is already a plot
        if len(self.canvas) > 0:
            start = self.options.offset_start
            end = self.options.offset_end
            if rotate:
                self.unrotated_mds_data = self.mds_data
                self.mds_data = libMDS.pca(self.mds_data, 2)
                self.clusters = libCluster.labelPoints(self.mds_data, self.label_indices[start:-end])
            elif self.unrotated_mds_data is not None:
                self.mds_data = self.unrotated_mds_data
                self.unrotated_mds_data = None
                self.clusters = libCluster.labelPoints(self.mds_data, self.label_indices[start:-end])
            self.rmmpl()
            self.plotLabel(self.clusters)

    def remove_timestamps(self):
        if not self.clusters:
            self.errDialog("Please open datafile first")

        filename, _ = QFileDialog.getOpenFileName(self, 'Open File', os.getenv('HOME'))
        try:
            with open(filename) as file:
               onoff= list((filter(None, (line.rstrip() for line in file))))
        except IOError:
             self.errDialog(str("Could not Open" + filename))
             return
        if len(onoff) > 2:
             self.errDialog("only start and end timestamp are supported")
             return

        start = int(onoff[0])
        if len(onoff) == 2:
            end = int(onoff[1])
        if start > self.data.shape[0] or end > self.data.shape[0]:
            self.errDialog("timestamp exceeds nr of points")
            return
        self.rmmpl()
        self.data = self.data[start:-end]
        self.label_indices = self.label_indices[start:-end]
        self.time_offset = start
        self.calcMDS(self.data)

    def showOptions(self):
        rotate = self.options.rotate
        self.options.show()

    def mdsException(self, err_msg):
        self.errDialog(err_msg)

        #remove progress bar
        if self.progressBar:
            self.mplvl.removeWidget(self.progressBar)
            self.progressBar.deleteLater()
            self.progressBar = None










#Worker thread for calculation, so that the ui doesnt freeze and we can show a progress bar
class WorkerThread(QtCore.QThread):
    finished = QtCore.pyqtSignal('PyQt_PyObject')
    exception = QtCore.pyqtSignal('PyQt_PyObject')

    #initialize thread with the high dimensional data
    def __init__(self, data):
        QtCore.QThread.__init__(self)
        self.data = data

    #this function is called automatically as soon as the tread is started
    def run(self):
        mds_data = None
        #calculate mds transformed data
        try:
            mds_data = libMDS.mds(self.data)
        except MemoryError:
            self.exception.emit("Out of memory")
            self.exit()
        except:
            self.exception.emit("MDS transformation failed")
            self.exit()

        #send calculated data back to main thread
        self.finished.emit(mds_data)

        #exit thread
        self.exit()

class Options(QOptions, Ui_Options):
    rotate = False
    permutations = 10000
    offset_start = 0
    offset_end = 0
    exclude_labels = []

    doRotation = QtCore.pyqtSignal('PyQt_PyObject')
    recalculate = QtCore.pyqtSignal('PyQt_PyObject')
    def __init__(self, ):
        super(Options, self).__init__()
        self.setupUi(self)
        self.buttonBox.accepted.connect(self.setValues)
        self.buttonBox.rejected.connect(self.restoreValues)
        self.rotate = self.ck_rotate.isChecked()
        self.permutations = int(self.text_permutations.text())
    def setValues(self):
        recalculate = False

        if self.offset_start != int(self.text_offset_start.text()) or self.offset_end != int(self.text_offset_end.text()):
            recalculate = True
        if self.rotate != self.ck_rotate.isChecked() and recalculate == False:
            self.doRotation.emit(self.ck_rotate.isChecked())
        self.rotate = self.ck_rotate.isChecked()
        self.permutations = int(self.text_permutations.text())
        self.offset_start = int(self.text_offset_start.text())
        self.offset_end = int(self.text_offset_end.text())
        self.exclude_labels = self.text_exclude_labels.split(',')


    def restoreValues(self):
        self.ck_rotate.setChecked(self.rotate)
        self.text_permutations.setText(str(self.permutations))
        self.text_offset_start.setText(str(self.offset_start))
        self.text_offset_end.setText(str(self.offset_end))
        self.text_exclude_labels.setText(','.join(self.exclude_labels))





if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Main()

    main.show()
    sys.exit(app.exec_())


