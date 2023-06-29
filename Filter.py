import time
import torch
import matplotlib.pyplot as plt

class ParticleFilter():
    def __init__(self, num_particles):
        self.init_time = time.time()

        self.move_time = 0
        self.resample_time = 0

        self.num_particles = num_particles
        self.particles = torch.zeros([num_particles, 4], dtype = torch.float32).cuda()

        self.init_time = time.time() - self.init_time

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

    def resample(self, embeddings):

        self.resample_time -= time.time()

        first_embedding = embeddings[0]
        weights = embeddings - first_embedding

        weights *= weights
        weights = embeddings.sum(axis = 1)/embeddings.sum()

        new_indices = torch.multinomial(weights, weights.shape[0], replacement=True)
        self.particles = self.particles[new_indices]

        self.resample_time += time.time()

    def print_times(self):
        print ("init time : ", self.init_time)
        print ("move time : ", self.init_time)
        print ("resample time : ", self.init_time)

    def draw_particles(self):
        xs = self.particles[:,0]
        ys = self.particles[:,1]
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
        plt.scatter(xs, ys)
        plt.show()


        
