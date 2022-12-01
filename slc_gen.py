# -*- coding:utf-8 -*-

##############################################
############### BY Caplab Team ###############
##############################################
#              Date: 11/21/22                #
##############################################

from pathlib import Path
import numpy as np
import pydicom
import png
import scipy.ndimage as ndimage
import math

def imgResample(point3d,hu_point3d,dcm_2elm,spacing=[1,1,1]):
    """ This function resize the images to 1x1x1 """

    print("Resampling process...")
    
    pix_sx,pix_sy,pix_th=dcm_2elm[0].PixelSpacing[0],dcm_2elm[0].PixelSpacing[1],abs(dcm_2elm[0].ImagePositionPatient[2]-dcm_2elm[1].ImagePositionPatient[2]) # abs(...) = slice thickness

    dim_arr=np.array([pix_sx,pix_sy,pix_th])
    sp=np.array(spacing)

    resize_fact=dim_arr/sp
    point3d_shape=np.array(list(point3d.shape))

    obj3d_size=point3d_shape*resize_fact
    obj3d_size=np.round(obj3d_size)
   
    resize_fact=obj3d_size/point3d_shape
      
    rsmpl_point3d=ndimage.interpolation.zoom(point3d,resize_fact)
    rsmpl_point3d=np.minimum(rsmpl_point3d,65535) # change all the values greater than 65535 to 65535
          
    rsmpl_hu_point3d=ndimage.interpolation.zoom(hu_point3d,resize_fact)
    rsmpl_hu_point3d=np.minimum(rsmpl_hu_point3d,65535) 

    print("Done!")
    
    return rsmpl_point3d,rsmpl_hu_point3d

def sagBones(point3d,hu_point3d,min_hu=500,max_hu=1900):
    shape=point3d.shape
    img=np.zeros((shape[1],shape[2]))
    boolean=False
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if(hu_point3d[i,j,k]>=min_hu and hu_point3d[i,j,k]<=max_hu):
                    if(img[j,k]==0):
                        img[j,k]=point3d[i,j,k]

    return img;

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
            array=point3d[i,:,:]
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
        hu_point3d=np.zeros(shape,dtype="float64")
        
        for i in range(0,self.nbr_axslices):

            pixel_array=np.maximum(slices[i].pixel_array,0).astype(float) # For compatibility purpose
            pixel_array=(pixel_array/slices[i].pixel_array.max())*65535 # We have used the value 65535 instead of 255 because medical images are stored with 2 bytes of gray shades. Using 255 will cause data loss.
            pixel_array=np.uint32(pixel_array)

            point2d=pixel_array
            point3d[:,:,i]=point2d
            hu_point3d[:,:,i]=(slices[i].pixel_array*slices[i].RescaleSlope)+slices[i].RescaleIntercept

        tmp=imgResample(point3d,hu_point3d.astype("int32"),[slices[0],slices[1]],[1,1,1])

        self.point3d=tmp[0]
        self.hu_point3d=tmp[1]
            
        self.ax3d=Ax3D(self.point3d)
        self.cor3d=Cor3D(self.point3d)
        self.sag3d=Sag3D(self.point3d)

        
    def saveSlices(self):
        """ Save slices in separate directory depending of the axis """
        cur=Path.cwd()
        
        ax=cur/"axial slices"
        cor=cur/"coronal slices"
        sag=cur/"sagital slices"

        nbr=0

        print("Saving slices...",end=" ")
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
                    
            print("Done!\n\nSlice(s) saved: {}!\n".format(nbr))
            
        except:
            print("Error while creating the directories for images.\nCheck that you have the permission to write on the directory\n")

class GenView(object):
    """ This class allow to create multiple different slice views """

    def __init__(self,arr3d,min_hu=500,max_hu=1900): # 500-1900 ---> hu range for cortical bones
        """ 
        arr3d: Gen3dArr object
        min_hu: minimum hu value for the targeted element of the body
        max_hu: maximum hu value for the targeted element of the body
        
        Note: Default min_hu and max_hu correspond to hu range for cortical bones
        
        """
        
        shape=list(arr3d.point3d.shape)
        x,y,z=shape.pop(0),shape.pop(1),shape.pop(2)

        if(x-y>=0):
            img_shape=(round(math.sqrt(math.pow(x,2)+math.pow(z,2))),)*2
        else:
            img_shape=(round(math.sqrt(math.pow(y,2)+math.pow(z,2))),)*2

        img=np.zeros(img_shape,dtype="uint32")
        img_arr=[]

        point3d=arr3d.point3d
        
