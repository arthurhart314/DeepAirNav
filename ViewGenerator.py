from MapLoader import MapVector, Map
import random
import numpy as np
import cv2
import os

class MapView(MapVector):
    def __init__(self):
        super().__init__()
        self.rpy = {'roll' : 0,     'pitch' : 0, 'yaw' : 0}

    def set_rpy(self, *args):
        if len(args) == 1:
            self.rpy = args[0]
        else:
            self.rpy = {'roll' : args[0], 'pitch' : args[1], 'yaw' : args[2]}


    def set_xyz_rpy(self, *args):
        if len(args) == 2:
            self.set_xyz(args[0])
            self.set_rpy(args[1])
        else:
            self.set_xyz(args[0], args[1], args[2])
            self.set_rpy(args[3], args[4], args[5])


    def set_lla_rpy(self, *args):
        if len(args) == 2:
            self.set_lla(args[0])
            self.set_rpy(args[1])
        else:
            self.set_lla(args[0], args[1], args[2])
            self.set_rpy(args[3], args[4], args[5])


    @staticmethod
    def from_xyz_rpy(*args):
        mview = MapView()
        mview.set_xyz_rpy(*args)
        return mview


    @staticmethod
    def from_lla_rpy(*args):
        mview = MapView()
        mview.set_lla_rpy(*args)
        return mview


    @staticmethod
    def from_mvec_rpy(*args):
        mvec = args[0]
        
        if len(args) == 2:
            rpy = args[1]
        else:
            rpy = {'roll' : args[1], 'pitch' : args[2], 'yaw' : args[3]}
        
        return MapView.from_lla_rpy(mvec.lla, rpy)

    ### Operators ####


    def __repr__(self):
        string = super().__repr__()
        string += '\n'
        string += 'rpy : ' + str(self.rpy['roll']) + ' ' + str(self.rpy['pitch']) + ' ' + str(self.rpy['yaw'])
        return string
    

    def __add__(self, other):
        
        xyz = {'x' : self.xyz['x'] + other.xyz['x'] , 'y' : self.xyz['y'] + other.xyz['y'], 'z' : self.xyz['z'] + other.xyz['z']}
        
        rpy = {'roll' : self.rpy['roll'] + other.rpy['roll'] , 'pitch' : self.rpy['pitch'] + other.rpy['pitch'], 'yaw' : self.rpy['yaw'] + other.rpy['yaw']}


        return self.from_xyz_rpy(xyz, rpy)
    
    def __sub__(self, other):

        xyz = {'x' : self.xyz['x'] - other.xyz['x'] , 'y' : self.xyz['y'] - other.xyz['y'], 'z' : self.xyz['z'] - other.xyz['z']}

        rpy = {'roll' : self.rpy['roll'] - other.rpy['roll'] , 'pitch' : self.rpy['pitch'] - other.rpy['pitch'], 'yaw' : self.rpy['yaw'] - other.rpy['yaw']}


        return self.from_xyz_rpy(xyz, rpy)
    
    def __rmul__(self, other):

        xyz = {'x' : self.xyz['x'] * other , 'y' : self.xyz['y'] * other, 'z' : self.xyz['z'] * 1}

        #rpy = {'roll' : self.rpy['roll'] * other , 'pitch' : self.rpy['pitch'] * other, 'yaw' : self.rpy['yaw'] * other}
        rpy = self.rpy

        return self.from_xyz_rpy(xyz, rpy)
    
    def __mul__(self, other):

        xyz = {'x' : self.xyz['x'] * other , 'y' : self.xyz['y'] * other, 'z' : self.xyz['z'] * 1}

        #rpy = {'roll' : self.rpy['roll'] * other , 'pitch' : self.rpy['pitch'] * other, 'yaw' : self.rpy['yaw'] * other}
        rpy = self.rpy

        return self.from_xyz_rpy(xyz, rpy)
    
    def __div__(self, other):

        xyz = {'x' : self.xyz['x'] / other , 'y' : self.xyz['y'] / other, 'z' : self.xyz['z'] / 1}

        #rpy = {'roll' : self.rpy['roll'] / other , 'pitch' : self.rpy['pitch'] / other, 'yaw' : self.rpy['yaw'] / other}
        rpy = self.rpy

        return self.from_xyz_rpy(xyz, rpy)
    
    def __eq__(self, other):
        return self.lla['lat'] == other.lla['lat'] and self.lla['lon'] == other.lla['lon'] and self.lla['alt'] == other.lla['alt']
    
    def __ne__(self, other):
        return not (self == other)

    

class Viewer():
    def __init__(self, map, views_folder_name = './Views', greyscale = False):
        self.views_folder_name = views_folder_name
        self.map = map
        self.greyscale = greyscale


    def x_rotation_matrix(self, theta):
        RX = np.array([[1,            0,             0, 0],
                       [0,np.cos(theta),-np.sin(theta), 0],
                       [0,np.sin(theta), np.cos(theta), 0],
                       [0,            0,             0, 1]])
        return RX
    
    def y_rotation_matrix(self, theta):
        RY = np.array([[ np.cos(theta), 0, np.sin(theta), 0],
                      [              0, 1,             0, 0],
                      [ -np.sin(theta), 0, np.cos(theta), 0],
                      [              0, 0,             0, 1]])
        return RY
    
    def z_rotation_matrix(self, theta):
        RZ = np.array([[ np.cos(theta), -np.sin(theta), 0, 0],
                       [ np.sin(theta),  np.cos(theta), 0, 0],
                       [             0,              0, 1, 0],
                       [             0,              0, 0, 1]])
        return RZ
    
    def intrinsic_matrix(self, f, h, w):
        # Camera intrinsic matrix
        K = np.array([[f, 0, w/2, 0],
                      [0, f, h/2, 0],
                      [0, 0,   1, 0]])
        return K
    
    def inverse_intrinsic_matrix(self, K):
        Kinv = np.zeros((4,3))
        Kinv[:3,:3] = np.linalg.inv(K[:3,:3])*K[0, 0]
        Kinv[-1,:] = [0, 0, 1]

        return Kinv
    
    def translation_matrix(self, x, y, z):
        T = np.array([[1, 0, 0, x],
                      [0, 1, 0, y],
                      [0, 0, 1, z],
                      [0, 0, 0, 1]])
        
        return T
    
    def img_view(self, src, x, y, z, rx, ry, rz, f_rs = 952.828, f_sat = 1e5):

        dst = np.zeros_like(src)
        h, w = src.shape[:2]

        #rx, ry, rz = 0, 0, 0
        #x , y,  z  = 0, 0, 100

        # Satellite intrinsic matrix
        
        K_sat = self.intrinsic_matrix(f_sat, h, w)
        K_sat_inv = self.inverse_intrinsic_matrix(K_sat)

        K_cam = self.intrinsic_matrix(f_rs, h, w)


        # Composed rotation matrix
        R     = np.linalg.multi_dot([ self.x_rotation_matrix(rx) , self.y_rotation_matrix(ry) , self.z_rotation_matrix(rz) ])
        T     = self.translation_matrix(x, y, z - f_sat)

        # Overall homography matrix
        H     = np.linalg.multi_dot([K_cam, R, T, K_sat_inv])

        # Apply matrix transformation
        im = np.zeros_like(src)
        im = cv2.warpPerspective(src, H, (w, h), im, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, 0)

        # Show the image
        return im
    

    def view_path_from_mview(self, mview):
        return self.views_folder_name + '/' + self.filename_from_mview(mview)


    #@staticmethod
    def filename_from_mview(self, mview):
        
        file_name = 'VIEW-'
        file_name += str(mview.lla['lat']  ).replace('.', '_').replace('-', 'n') + '-'
        file_name += str(mview.lla['lon']  ).replace('.', '_').replace('-', 'n') + '-'
        file_name += str(mview.lla['alt']  ).replace('.', '_').replace('-', 'n') + '-'
        file_name += str(mview.rpy['roll'] ).replace('.', '_').replace('-', 'n') + '-'
        file_name += str(mview.rpy['pitch']).replace('.', '_').replace('-', 'n') + '-'
        file_name += str(mview.rpy['yaw']  ).replace('.', '_').replace('-', 'n') + '.jpeg'

        return file_name
    
    def mview_from_filename(self, file_name):

        _, lat, lon, alt, roll, pitch, yaw = file_name[:-5].replace('-', '=').replace('_', '.').replace('n', '-').split('=')

        lla = {'lat' : float(lat), 'lon' : float(lon), 'alt' : float(alt)}
        rpy = {'roll' : float(roll), 'pitch' : float(pitch), 'yaw' : float(yaw)}
        
        mview = MapView.from_lla_rpy(lla, rpy)

        return mview
    

    def view_is_in_folder(self, mview):
        exists = os.path.exists(self.view_path_from_mview(mview))
        return exists
    

    def get_view_from_folder(self, mview):
        img = img = cv2.imread(self.view_path_from_mview(mview))
        return img

    def get_view(self, mview):

        if not self.view_is_in_folder(mview):
            img = self.generate_view(mview)

        img = self.get_view_from_folder(mview)
        
        return img


    def generate_view(self, mview):

        id = self.map.region_id_from_map_vec(mview)
        region_img = self.map.get_region(id)

        #print (id)


        h, w = region_img.shape[:2]
        region_mvec = self.map.region_position_from_id(id)


        #print (mview)
        #print (MapView.from_mvec_rpy(region_mvec, 0, 0, 0))
        pix_mview = (mview - MapView.from_mvec_rpy(region_mvec, 0, 0, 0))*(w/self.map.region_length)
        pix_mview.xyz['y']  = -1 * pix_mview.xyz['y']


        circle_h, circle_w = int(h/2 + pix_mview.xyz['y']), int(w/2 + pix_mview.xyz['x'])
        circle_r, circle_t = 100, 20
        
        #region_img_circle = cv2.circle(region_img, (circle_w, circle_h), circle_r, (0,0,255), circle_t)
        #cv2.line(region_img_circle, (int(circle_w - circle_r), circle_h), (int(circle_w + circle_r), circle_h), (0,0,255), 20)
        #cv2.line(region_img_circle, (circle_w, int(circle_h - circle_r)), (circle_w, int(circle_h + circle_r)), (0,0,255), 20)

        satellite_f = 100000
        img = self.img_view(region_img, -pix_mview.xyz['x'],   -pix_mview.xyz['y'],     pix_mview.xyz['z'], 
                                        pix_mview.rpy['roll'], pix_mview.rpy['pitch'], pix_mview.rpy['yaw'])
        
        img = cv2.resize(img, (500, 500),interpolation = cv2.INTER_AREA)
        
        if self.greyscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #mask = self.img_view(img             ,  pix_mview.xyz['x'],    pix_mview.xyz['y'],     pix_mview.xyz['z'], 
        #                                        -pix_mview.rpy['roll'], -pix_mview.rpy['pitch'], -pix_mview.rpy['yaw']) ### TODO!
        
        cv2.imwrite(self.view_path_from_mview(mview), img)

    def delete_views(self):
        os.system('rm -r ' + self.views_folder_name)
        os.system('mkdir ' + self.views_folder_name)





def view_sampler(n, xyz_min, xyz_max, rpy_min, rpy_max, offline = False, map = None):
    mview_arr = []
    
    while len(mview_arr) < n:
        x = random.uniform(xyz_min['x'], xyz_max['x'])
        y = random.uniform(xyz_min['y'], xyz_max['y'])
        z = random.uniform(xyz_min['z'], xyz_max['z'])

        roll  = random.uniform(rpy_min['roll'] , rpy_max['roll'] )
        pitch = random.uniform(rpy_min['pitch'], rpy_max['pitch'])
        yaw   = random.uniform(rpy_min['yaw']  , rpy_max['yaw']  )

        mview = MapView.from_xyz_rpy(x, y, z, roll, pitch, yaw)

        
        if offline and not map.region_is_in_folder(map.region_id_from_map_vec(mview)):
            pass
        else:
            mview_arr.append(mview)

    return mview_arr

def view_grid(n_xyz, xyz_min, xyz_max, n_rpy, rpy_min, rpy_max):

    X = np.linspace(xyz_min['x'], xyz_max['x'], n_xyz['x'])
    Y = np.linspace(xyz_min['y'], xyz_max['y'], n_xyz['y'])
    Z = np.linspace(xyz_min['z'], xyz_max['z'], n_xyz['z'])
    
    ROLL  = np.linspace(rpy_min['roll'] , rpy_max['roll'] , n_rpy['roll'] )
    PITCH = np.linspace(rpy_min['pitch'], rpy_max['pitch'], n_rpy['pitch'])
    YAW   = np.linspace(rpy_min['yaw']  , rpy_max['yaw']  , n_rpy['yaw']  )

    mview_arr = []

    for x in X:
        for y in Y:
            for z in Z:
                for roll in ROLL:
                    for pitch in PITCH:
                        for yaw in YAW:
                            mview = MapView.from_xyz_rpy(x, y, z, roll, pitch, yaw)
                            mview_arr.append(mview)

    return mview_arr


def get_mview_arr_from_folder(path):
    pass



