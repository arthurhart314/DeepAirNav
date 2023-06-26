import cv2
import numpy as np
import requests
import time

import math
import multiprocessing
from tqdm import tqdm


import pymap3d as pm

import os
import cv2

import BingMaps as bm

import matplotlib.pyplot as plt


class MapVector():
    def __init__(self):
        
        self.xyz = {'x' : 0, 'y' : 0, 'z' : 0}

        self.lla = {'lat' : 0, 'lon' : 0, 'alt' : 0}

        self.lla_0 = {'lat' : 31.884542, 'lon' : 34.961650, 'alt' : 0}


    def xyz_from_lla(self):
        self.xyz['x'], self.xyz['y'], self.xyz['z'] = pm.geodetic2enu(self.lla['lat'], self.lla['lon'], self.lla['alt'], self.lla_0['lat'], self.lla_0['lon'], self.lla_0['alt'])


    def lla_from_xyz(self):
        self.lla['lat'], self.lla['lon'], self.lla['alt'] = pm.enu2geodetic(self.xyz['x'], self.xyz['y'], self.xyz['z'], self.lla_0['lat'], self.lla_0['lon'], self.lla_0['alt'])

    def set_lla_0(self, lla_0):
        self.lla_0 = lla_0
        self.xyz_from_lla()

    def set_lla(self, *args):
        if len(args) == 1:
            self.lla = args[0]
        else:
            self.lla = {'lat' : args[0], 'lon' : args[1], 'alt' : args[2]}

        self.xyz_from_lla()

    def set_xyz(self, *args):
        if len(args) == 1:
            self.xyz = args[0]
        else:
            self.xyz = {'x' : args[0], 'y' : args[1], 'z' : args[2]}
            
        self.lla_from_xyz()

    @staticmethod
    def from_lla(*args):
        mvec = MapVector()
        mvec.set_lla(*args)
        return mvec

    @staticmethod
    def from_xyz(*args):
        mvec = MapVector()
        mvec.set_xyz(*args)
        return mvec
    

    ## Map Vector Operators :

    def __add__(self, other):
        
        xyz = {'x' : self.xyz['x'] + other.xyz['x'] , 'y' : self.xyz['y'] + other.xyz['y'], 'z' : self.xyz['z'] + other.xyz['z']}

        return self.from_xyz(xyz)
    
    def __sub__(self, other):

        xyz = {'x' : self.xyz['x'] - other.xyz['x'] , 'y' : self.xyz['y'] - other.xyz['y'], 'z' : self.xyz['z'] - other.xyz['z']}

        return self.from_xyz(xyz)
    
    def __rmul__(self, other):

        xyz = {'x' : self.xyz['x'] * other , 'y' : self.xyz['y'] * other, 'z' : self.xyz['z'] * other}

        return self.from_xyz(xyz)
    
    def __mul__(self, other):

        xyz = {'x' : self.xyz['x'] * other , 'y' : self.xyz['y'] * other, 'z' : self.xyz['z'] * other}

        return self.from_xyz(xyz)
    
    def __div__(self, other):

        xyz = {'x' : self.xyz['x'] / other , 'y' : self.xyz['y'] / other, 'z' : self.xyz['z'] / other}

        return self.from_xyz(xyz)
    
    def __repr__(self):
        #string = self.__class__.__name__
        #string += '\n'
        string = ''
        string += 'xyz : ' + str(self.xyz['x']) + ' ' + str(self.xyz['y']) + ' ' + str(self.xyz['z'])
        string += '\n'
        string += 'lla : ' + str(round(self.lla['lat'], 10)) + ' ' + str(round(self.lla['lon'], 10)) + ' ' + str(round(self.lla['alt'], 3))
        return string
    
    def __eq__(self, other):
        return self.lla['lat'] == other.lla['lat'] and self.lla['lon'] == other.lla['lon'] and self.lla['alt'] == other.lla['alt']
    
    def __ne__(self, other):
        return not (self == other)



class Map():
    def __init__(self, maps_folder_name):
        self.maps_folder_name = maps_folder_name

        self.lla_0 = {'lat' : 31.884542, 'lon' : 34.961650, 'alt' : 0}

        self.region_length = 1000
        self.region_distance = 500

        pass

    def regions_ids_from_map_vec_arr(self, mvec_arr):
        """Returns two arrays:
           * An array of all regions that will be used
           * An array of regions ids for each position element"""
        ids_map = {}
        ids_arr = []

        for mvec in mvec_arr:
            id = self.region_id_from_map_vec(mvec)
            ids_map[id] = ids_map.get(id, 0) + 1
            ids_arr.append(id)

        return ids_arr, ids_map
        

    def region_id_from_map_vec(self, mvec):
        """Returns and id of the region in the string format : n2-3"""
        #mvec.set_lla_0(self.lla_0)
        X_n = round(mvec.xyz['x']/self.region_distance )
        Y_n = round(mvec.xyz['y']/self.region_distance )

        return  (str(round(X_n)) + '=' + str(round(Y_n))).replace('-', 'n').replace('=', '-')


    def region_position_from_id(self, id):
        """Returns an instance MapVector positioned at the center of the given rigion"""
        arr = id.replace('-', '=').replace('n', '-').split('=')
        X_n, Y_n = float(arr[0]), float(arr[1])
        mvec = MapVector.from_xyz(X_n*self.region_distance, Y_n*self.region_distance, 0)

        return mvec
    
    
    def region_path_from_id(self, id):

        file_name = id + '.png'
        full_path = self.maps_folder_name + '/' + file_name

        return full_path

    
    def region_is_in_folder(self, id):

        exists = os.path.exists(self.region_path_from_id(id))

        return exists
    
    def get_region_from_folder(self, id):

        img_path = self.region_path_from_id(id)

        img = cv2.imread(img_path)

        return img


    def load_region_from_bing(self, id):
        """Downloads the image of self.region_length**2 area region from Bing maps and stores in map folder"""
        mvec = self.region_position_from_id(id)
        try:
            bm.load(mvec.lla['lat'], mvec.lla['lon'], self.region_length, self.region_path_from_id(id), self.maps_folder_name + '/tmp')
        except:
            print ("Could not load region from bing maps! \n")

        

    def get_region(self, id):
        """Returns the image image of self.region_length**2 area region"""
        if not self.region_is_in_folder(id):
            self.load_region_from_bing(id)

        img = self.get_region_from_folder(id)

        return img
    
    def display_region(self, id):
        img = self.get_region(id)
        plt.title(id)
        plt.imshow(img)
        plt.show()

    def display_mvec_arr_locations(self, region_id, mvec_arr, color_arr = [], rect_start = False, rect_end = False):
        
        region_img = self.get_region(region_id)
        region_mvec = self.region_position_from_id(region_id)

        for i in range(len(mvec_arr)):

            h, w = region_img.shape[:2]

            pix_mvec = (mvec_arr[i] - region_mvec)*(w/self.region_length)
            pix_mvec.xyz['y']  = -1 * pix_mvec.xyz['y']

            circle_h, circle_w = int(h/2 + pix_mvec.xyz['y']), int(w/2 + pix_mvec.xyz['x'])
            circle_r, circle_t = 5, 5
        
            if len(color_arr) != 0:
                color = color_arr[i]
            else:
                color = (0,0,255)

            #print (color)

            cv2.circle(region_img, (circle_w, circle_h), circle_r, color, circle_t)
            cv2.line  (region_img, (int(circle_w - circle_r), circle_h), (int(circle_w + circle_r), circle_h), color, 20)
            cv2.line  (region_img, (circle_w, int(circle_h - circle_r)), (circle_w, int(circle_h + circle_r)), color, 20)

        if rect_start and rect_end:
                cv2.rectangle(region_img, rect_start, rect_end, 10)

        return region_img

    def delete_maps(self):
        os.system('rm -r ' + self.maps_folder_name)
        os.system('mkdir ' + self.maps_folder_name)


