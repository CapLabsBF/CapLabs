# -*- using:utf-8 -*-

##############################################
############### BY Caplab Team ###############
##############################################
#              Date: 11/21/22                #
##############################################

from pathlib import Path
import numpy as np
import pydicom
import png

class Ax3D(object):
    """ Creates an array of pixels that are linked to the axial axis """

    def __init__(self,point3d):
        """"""
        shape=list(point3d.shape)
        
        self.nbr_slices=shape.pop()

        array=np.zeros(shape)
        list_array=[]

        for i in range(0,self.nbr_slices):
            array=point3d[:,:,i]
            list_array.append(array)

        self.pixel_array_list=list_array

class Cor3D(object):
    """ Creates an array of pixels that are linked to the coronal axis """

    def __init__(self,point3d):
        """"""
        shape=list(point3d.shape)
        
        self.nbr_slices=shape.pop(1)

        array=np.zeros(shape)
        list_array=[]

        for i in range(0,self.nbr_slices):
            array=point3d[:,i,:]
            list_array.append(array)

        self.pixel_array_list=list_array

class Sag3D(object):
    """ Creates an array of pixels that are linked to the coronal axis """

    def __init__(self,point3d):
        """"""
        shape=list(point3d.shape)

        self.nbr_slices=shape.pop(0)

        array=np.zeros(shape)
        list_array=[]

        for i in range(0,self.nbr_slices):
            array=point3d[i,:,:].T # The array must be Transposed
            list_array.append(array)

        self.pixel_array_list=list_array
        
class Gen3dArr(object):
    """This class will allow you to create an array of 3d points."""

    def __init__(self,slices: pydicom.filebase.DicomFileLike):
        """ """
        self.nbr_axslices=len(slices)
        shape=list(slices[0].pixel_array.shape)
        shape.append(self.nbr_axslices)
        point3d=np.zeros(shape,dtype="uint32") # We are considering the same shape for all the slices
        
        for i in range(0,self.nbr_axslices):
            pixel_array=np.maximum(slices[i].pixel_array,0).astype(float) # For compatibility purpose
            pixel_array=(pixel_array/slices[i].pixel_array.max())*65535 # We have used the value 65535 instead of 255 because medical images are stored with 2 bytes of gray shades. Using 255 will cause data loss.
            pixel_array=np.uint32(pixel_array)

            point2d=pixel_array
            point3d[:,:,i]=point2d
        
        self.ax3d=Ax3D(point3d)
        self.cor3d=Cor3D(point3d)
        self.sag3d=Sag3D(point3d)

        self.point3d=point3d
        
    def saveSlices(self):
        """ Save slices in separate directory depending of the axis """
        cur=Path.cwd()
        
        ax=cur/"axial slices"
        cor=cur/"coronal slices"
        sag=cur/"sagital slices"

        nbr=0

        try:
            ax.mkdir()
            cor.mkdir()
            sag.mkdir()

            for i in range(0,self.ax3d.nbr_slices):
                try:
                    img=png.from_array(self.ax3d.pixel_array_list[i],"L;16")
                    img.save("axial slices/ax_img"+str(i)+".png")
                    nbr=nbr+1
                except:
                    print("Error while saving {}".format("ax_img"+str(i)+".png"))

            for i in range(0,self.cor3d.nbr_slices):
                try:
                    img=png.from_array(self.cor3d.pixel_array_list[i],"L;16")
                    img.save("coronal slices/cor_img"+str(i)+".png")
                    nbr=nbr+1
                except:
                    print("Error while saving {}".format("cor_img"+str(i)+".png"))
                          
            for i in range(0,self.sag3d.nbr_slices):
                try:
                    img=png.from_array(self.sag3d.pixel_array_list[i],"L;16")
                    img.save("sagital slices/sag_img"+str(i)+".png")
                    nbr=nbr+1
                except:
                    print("Error while saving {}".format("sag_img"+str(i)+".png"))
                    
            print("\nSlice(s) saved: {}!\n".format(nbr))
            
        except:
            print("Error while creating the directories for images.\nCheck that you have the permission to write on the directory\n")
