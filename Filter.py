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
        self.particles[:,3] += displacement_rand[:,3]

        self.move_time += time.time()
        self.move_n += 1


    def to_geodetic(self, lla0):

        self.coord_transform_time -= time.time()

        lat, lon, alt = pm.ned2geodetic(self.particles[:,0].cpu().numpy(), 
                                        self.particles[:,1].cpu().numpy(),
                                        self.particles[:,2].cpu().numpy(),
                                        lla0[0], lla0[1], lla0[2])
                                        
        self.geo_particles[:,0] = lat
        self.geo_particles[:,1] = lon
        self.geo_particles[:,2] = alt
        self.geo_particles[:,3] = self.particles[:,3].cpu().numpy()

        self.coord_transform_time += time.time()

        return self.geo_particles


    def resample(self, embeddings):

        self.resample_time -= time.time()

        first_embedding = embeddings[0]
        weights = embeddings - first_embedding

        weights *= weights
        weights = embeddings.sum(axis = 1)/embeddings.sum()

        new_indices = torch.multinomial(weights, weights.shape[0], replacement=True)
        self.particles = self.particles[new_indices]

        self.resample_time += time.time()
        self.resample_n += 1


    def print_times(self):
        print ("init time : ", self.init_time)
        print ("move time : ", self.move_time/self.move_n)
        print ("resample time : ", self.resample_time/self.resample_n)
        print ("coord transform time : ", self.coord_transform_time/self.move_n)


    def draw_particles(self):
        xs = self.particles[:,0].cpu().numpy()
        ys = self.particles[:,1].cpu().numpy()
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
        plt.scatter(xs, ys, s=2)
        plt.show()


        
