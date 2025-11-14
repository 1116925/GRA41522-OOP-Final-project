'''
    A module that holds the DataLoader class used for loading datasets into the training models.
'''

class DataLoader:
    '''
    A DataLoader class for loading datasets and making them \n
    available to the ML model.
    '''
    def __init__(self, dataset_name: str):
        '''
        Constructor for DataLoader class.

        @param dataset_name: Name of the dataset to load.
        '''
        self._dataset_name = dataset_name
