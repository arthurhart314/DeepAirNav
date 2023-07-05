import time
import torch
import matplotlib.pyplot as plt
import numpy as np

import pymap3d as pm

class ParticleFilter():
    def __init__(self, num_particles):
        self.init_time = -time.time()

        self.move_time = 0
        self.move_n = 0
        self.resample_time = 0
        self.resample_n = 0

        self.coord_transform_time = 0

        self.probas = None
        self.dist = None

        self.num_particles = num_particles
        self.particles = torch.zeros([num_particles, 4], dtype = torch.float32).cuda()
        self.geo_particles = np.zeros([num_particles, 4], dtype = np.float64)

        self.init_time += time.time()

    def move(self, displacement_val, displacement_var):

        self.move_time -= time.time()

        displacement_val = displacement_val[None, :].repeat(self.num_particles, 1)
        displacement_var = displacement_var[None, :].repeat(self.num_particles, 1)


        displacement_rand= torch.normal(displacement_val, displacement_var)
        
        self.particles[:,0] += displacement_rand[:,0]*torch.cos(self.particles[:,3]) - \
                               displacement_rand[:,1]*torch.sin(self.particles[:,3])
        self.particles[:,1] += displacement_rand[:,0]*torch.sin(self.particles[:,3]) + \
                               displacement_rand[:,1]*torch.cos(self.particles[:,3])
        
        self.particles[:,2] += displacement_rand[:,2]

        self.particles[:,2] = torch.relu((self.particles[:,2]))

        self.particles[:,3] += displacement_rand[:,3]

        self.move_time += time.time()
        self.move_n += 1

    def sample_particles(self, x_range, y_range, z_range, yaw_range):

        x = np.random.uniform(x_range[0], x_range[1], self.num_particles)
        y = np.random.uniform(y_range[0], y_range[1], self.num_particles)
        z = np.random.uniform(z_range[0], z_range[1], self.num_particles)
        yaw = np.random.uniform(yaw_range[0], yaw_range[1], self.num_particles)

        self.particles = torch.tensor(np.array([x, y, z, yaw]).T, dtype = torch.float32).cuda()

        return self.particles


    def to_geodetic(self, lla0 =  [31.884542, 34.961650, 0]):

        self.coord_transform_time -= time.time()

        lat, lon, alt = pm.enu2geodetic(self.particles[:,0].cpu().numpy(), 
                                     -1*self.particles[:,1].cpu().numpy(),
                                        self.particles[:,2].cpu().numpy(),
                                        lla0[0], lla0[1], lla0[2])
                                        
        self.geo_particles[:,0] = lat
        self.geo_particles[:,1] = lon
        self.geo_particles[:,2] = alt
        self.geo_particles[:,3] = self.particles[:,3].cpu().numpy()

        self.coord_transform_time += time.time()

        return self.geo_particles

    @torch.no_grad() ## check resampling!!

    def calculate_probabilities(self, embeddings, target_embedding):
        dist = (embeddings - target_embedding)*(embeddings - target_embedding)
        dist = dist.sum(axis = 1)

        dist_np = dist.cpu().numpy()

        self.dist = dist

        probas = 1 - (dist - dist.min())/(dist.max() - dist.min())

        self.probas = probas

    def resample(self):

        self.resample_time -= time.time()
        
        new_indices = torch.multinomial(self.probas, self.probas.shape[0], replacement=True)

        self.particles = self.particles[new_indices]
        self.dist = self.dist[new_indices]

        self.resample_time += time.time()
        self.resample_n += 1


    def print_times(self):
        print ("init time : ", self.init_time)
        print ("move time : ", self.move_time/self.move_n)
        print ("resample time : ", self.resample_time/self.resample_n)
        print ("coord transform time : ", self.coord_transform_time/self.move_n)


    def colors_from_probas(self):

        p = self.probas.cpu().numpy()

        color_arr = (p - p.min())/(p.max() - p.min())

        color_arr = color_arr.astype(np.float32)
        color_arr = [(float(1.0 - c), 0, float(c)) for c in color_arr]

        return color_arr
    
    def get_mean_std(self):


        mean = self.particles.mean(axis = 0)
        std = self.particles.std(axis = 0)

        self.to_geodetic()

        mean_lla = self.geo_particles.mean(axis = 0)
        std_lla = self.geo_particles.std(axis = 0)



        return mean , std, mean_lla, std_lla


    def draw_particles(self, region_id): ## test drawing!!!!!
        xs = self.particles[:,0].cpu().numpy() + 500
        ys = -self.particles[:,1].cpu().numpy() + 500

        import cv2

        colors = self.colors_from_probas()


        img = cv2.imread("./Maps/0-0.png")
        img = cv2.resize(img, (1000, 1000))

        img = cv2.flip(img, 0)

        plt.imshow(img)

        plt.xlim([0, 1000])
        plt.ylim([0, 1000])

        plt.scatter(xs, ys, s=1, c='r', marker='o')
        plt.show()

    def draw_histograms(self):
        plt.hist(self.probas.cpu().numpy(), bins=100)
        plt.show()
        plt.hist(self.dist.cpu().numpy(), bins=100)
        plt.show()


        
