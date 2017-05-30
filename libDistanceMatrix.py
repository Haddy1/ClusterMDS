import numpy as np

class DistanceMatrix(np.ndarray):
    #Subclassing ndarray is a bit different from other python Classes - using new instead of init
    def __new__(cls, labels, array = None):
        if labels is None: return None

        if array is not None:
            self = array.view(cls)
        else:
            self = np.zeros((len(labels),len(labels))).view(cls)
        self.labels = list(labels)
        return self

    #This gets called by view function of numpy
    def __array_finalize__(self, obj):
        if obj is None: return
        self.labels = getattr(obj, 'labels', None)


    def strToIndex(self,s):
        index_start = self.labels.index(s)
        if index_start is None:
            raise IndexError("matrix does not contain " + str(s))
            return None

        index_stop = len(self.labels) - next(i for i, v in enumerate(reversed(self.labels), 1) if v == s)
        if index_stop == index_start:
            index = index_start
        else:
            index = slice(index_start, index_stop + 1)
        return index
    def convertSlice(self,sl):
        if isinstance(sl.start, str):
            #get first occurance of index.start in self.labels
            sl.start = self.labels.index(sl.start)
        if isinstance(sl.stop, str):
            #search for last occurance of index.stop in self.labels
            sl.stop = len(self.labels) - next(i for i, v in enumerate(reversed(self.labels), 1) if v == s) +1
        return sl



    #overwrites [] operator  for getting items
    def __getitem__(self, index):
        if index is None: return

        if isinstance(index, tuple):
            if len(index) > 2: raise IndexError("too many indices for array")
            first, second = index

            if isinstance(index[0], str):
                first = self.strToIndex(index[0])
            elif index[0] is slice:
                first = self.convertSlice(index[0])
            if isinstance(index[1], str):
                second = self.strToIndex(index[1])
            elif index[1] is slice:
                second = self.convertSlice(index[1])
            index = (first, second)

        elif isinstance(index, str):
            index = self.strToIndex(index)
        elif index is slice:
            index = self.convertSlice(index)


        #view converts super class ndarray to DistanceMatrix by calling __array_finalize__
        #view does not allocate new memory, simple changes the Class the memory is read as
        return super(DistanceMatrix, self).__getitem__(index).view(DistanceMatrix)

    #overwrites [] operator for setting items
    def __setitem__(self, index, item):
        if index is None: return

        if isinstance(index, tuple):
            if len(index) > 2: raise IndexError("too many indices for array")
            first, second = index

            if isinstance(index[0], str):
                first = self.strToIndex(index[0])
            elif index[0] is slice:
                first = self.convertSlice(index[0])
            if isinstance(index[1], str):
                second = self.strToIndex(index[1])
            elif index[1] is slice:
                second = self.convertSlice(index[1])
            index = (first, second)


        elif isinstance(index, str):
            index = self.strToIndex(index)
        elif index is slice:
            index = self.convertSlice(index)

        #no view and return here for some reason (would return None)
        super(DistanceMatrix, self).__setitem__(index, item)
