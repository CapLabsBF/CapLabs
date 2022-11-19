# -*- coding:utf-8 -*-

##############################################
############### BY Caplab Team ###############
##############################################
#              Date: 11/21/22                #
##############################################

from pathlib import Path
import glob
import pydicom

def loadData(path: str):
    """"""
    dcm_elm,slices=[],[]
    dir=Path(path)

    if(dir.is_dir()):

        path=path+"/*.dcm"
        
        print("Loading the files:")
        print("path: {}\n".format(path))
        
        path_list=glob.glob(path)
        
        for path in path_list:
            print("\t{}".format(path))
            try:
                dcm_elm.append(pydicom.dcmread(path))
            except:
                print("\nError while reading the file {}\n".format(path))
                return None
            
        print("\nDone! Number of files loaded: {}\n".format(len(path_list)))
        
        for elm in dcm_elm:
            if(hasattr(elm,"SliceLocation")):
                slices.append(elm)
                slices.sort(key= lambda s:s.SliceLocation) # Order the slices by their SliceLocation value

        print("Slice(s) skipped: {}".format(len(path_list)-len(slices)))
        
        return slices
    
    else:
        print("\nError while reading the path: {}\n".format(path))
        return None
        
