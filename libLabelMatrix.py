import numpy as np

class LabelMatrix(np.ndarray):
    #Subclassing ndarray is a little bit different from other python Classes - using new instead of init
    def __new__(cls, labels, array = None):
        if labels is None: return None

        if array is not None:
            self = array
        else:
            self = np.zeros((len(labels),len(labels))).view(cls)
        self.labels = list(labels)
        return self

    #This gets called by view function of numpy
    def __array_finalize__(self, obj):
        if obj is None: return
        self.labels = getattr(obj, 'labels', None)


    def __getitem__(self, index):
        if index is None: return

        if isinstance(index, str):
            index = self.labels.index(index)

        #view converts super class ndarray to LabelMatrix by calling __array_finalize__
        #view does not allocate new memory, simple changes the Class the memory is read as
        return super(LabelMatrix, self).__getitem__(index).view(LabelMatrix)

    def __setitem__(self, index, item):
        if index is None: return

        if isinstance(index, str):
            index = self.labels.index(index)

        #no view and return here for some reason (would return None)
        super(LabelMatrix, self).__setitem__(index, item)
