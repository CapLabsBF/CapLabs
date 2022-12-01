#!/usr/bin/python3

# -*- coding:utf-8 -*-

##############################################
############### BY Caplab Team ###############
##############################################
#              Date: 11/21/22                #
##############################################

import sys
from slc_gen import Gen3dArr
from dataLoader import loadData

# *---> Main

if(len(sys.argv) != 2):
    print("Needs one arguments\nSyntaxe: {} or\n\t{}".format("./script.py 'files_path'","python script.py 'file_path'"))
else:
    slices=loadData(sys.argv[1])
    if(slices != None):
        point3D=Gen3dArr(slices)
        #point3D.saveSlices()
    
# <---*
