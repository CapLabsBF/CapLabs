###################################
########                 ##########
#######   BY CapLabs    ###########     
########                 ##########
###################################


from pathlib import Path
from joblibrary import Loadjob
import joblib
import numpy as np

class BreastDataset():

    """
    Breast Dataset class
    """

    def __init__(
            self,
            transform=None,):

        pass

    def splitting(self,):
        self.direc = "BCFPP.jlb"
        """
        splitting the image dataset :
        3600 : total
        2520 : Training 70%
        540 : Validation 15%
        540 : Testing 15%
        """

        # Loading the dataset with joblib
        data = Loadjob(self.direc)
        #return data
        # The initial dataset has images, labels and masks objects
        # data is a tuple of arrays
        # images = data[0], labels = data[1], masks = data[2]
        # We do not need the masks        
        # Split calculations in the data_split.txt file
        
        train, trainlabels = np.append(data[0][0:1680], data[0][2400:3240], axis=0) , np.append(data[1][0:1680], data[1][2400:3240], axis=0)
        
        val, vallabels = np.append(data[0][1680:2040], data[0][3240:3420], axis=0), np.append(data[1][1680:2040], data[1][3240:3420], axis=0)
        
        test, testlabels = np.append(data[0][2040:2400], data[0][3420:3600], axis=0), np.append(data[1][2040:2400], data[1][3420:3600], axis=0)

        return [train, val, test], [trainlabels, vallabels, testlabels] 




        





        
