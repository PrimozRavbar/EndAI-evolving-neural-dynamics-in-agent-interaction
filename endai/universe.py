import numpy as np
from scipy import ndimage

class Universe():

    def __init__(self,size_universe):

        self.size_universe = size_universe

        #self.config = "open_field"
        self.config = "box_field"

        self.uni_mat = np.zeros((self.size_universe,self.size_universe,3))

    def update_uni_mat(self,pos_y,pos_x,color_value,hidden_value=0):

        self.uni_mat[int(pos_y),int(pos_x),:] = color_value

    def draw_edges(self):

        thickness = 3

        self.uni_mat[:,0:thickness,:] = [0,0,1]
        self.uni_mat[:,self.size_universe-thickness:self.size_universe,:] = [0,0,1]

        self.uni_mat[0:thickness,:,:] = [0,0,1]
        self.uni_mat[self.size_universe-thickness:self.size_universe,:,:] = [0,0,1]

    def add_object_to_uni(self,current_object):

        ob_size = current_object.body_size
        ob_color = current_object.body_color

        self.uni_mat[int(current_object.pos_y)-ob_size:int(current_object.pos_y)+ob_size,
                     int(current_object.pos_x)-ob_size:int(current_object.pos_x)+ob_size,:] += ob_color #[.7,.45,.100]

        #self.uni_mat[int(current_object.pos_y),int(current_object.pos_x),:] += [1,0,1] #[.7,.45,.100]

    def clear_previous_object (self,prev_object):

        ob_size = prev_object.body_size
        ob_color = prev_object.body_color

        self.uni_mat[int(prev_object.pos_y)-ob_size:int(prev_object.pos_y)+ob_size,
                     int(prev_object.pos_x)-ob_size:int(prev_object.pos_x)+ob_size,:] -= ob_color

        #self.uni_mat[int(prev_object.pos_y),int(prev_object.pos_x),:] -= [1,0,1]

    def add_inam_object_to_uni(self,current_object):

        ob_size_x = current_object.body_size_x
        ob_size_y = current_object.body_size_y
        ob_color = current_object.body_color

        self.uni_mat[int(current_object.pos_y)-ob_size_y:int(current_object.pos_y)+ob_size_y,
                     int(current_object.pos_x)-ob_size_x:int(current_object.pos_x)+ob_size_x,:] = \
        self.uni_mat[int(current_object.pos_y)-ob_size_y:int(current_object.pos_y)+ob_size_y,
             int(current_object.pos_x)-ob_size_x:int(current_object.pos_x)+ob_size_x,:] +  ob_color #[.7,.45,.100]



    def clear_previous_inam_object (self,prev_object):

        ob_size_x = current_object.body_size_x
        ob_size_y = current_object.body_size_y
        ob_color = current_object.body_color

        self.uni_mat[int(current_object.pos_y)-ob_size_y:int(current_object.pos_y)+ob_size_y,
                     int(current_object.pos_x)-ob_size_x:int(current_object.pos_x)+ob_size_x,:] = [0,0,0] #[.7,.45,.100]




    def gaussian_2d(self, x, y, mean, cov):
        """Generate a 2D Gaussian distribution."""
        inv_cov = np.linalg.inv(cov)
        diff = np.dstack((x - mean[0], y - mean[1]))
        exponent = -0.5 * np.einsum('...k,kl,...l->...', diff, inv_cov, diff)
        return np.exp(exponent)

    def create_energy_field(self, size_array, total_sum):
        # Create a zero array of size (size_array x size_array)
        array = np.zeros((size_array, size_array))

        # Create a meshgrid for the array
        x = np.linspace(0, size_array - 1, size_array)
        y = np.linspace(0, size_array - 1, size_array)
        x, y = np.meshgrid(x, y)

        # Generate 4 random means and very broad covariances
        means = [np.random.uniform(20, 80, 2) for _ in range(4)]
        covariances = [np.diag(np.random.uniform(200, 400, 2)) for _ in range(4)]  # Further increased covariance values for broader Gaussians

        # Add the 2D Gaussians to the array
        for mean, cov in zip(means, covariances):
            array += self.gaussian_2d(x, y, mean, cov)

        # Normalize the array to have the desired total sum
        array *= total_sum / np.sum(array)

        # Ensure the minimum value is at least 0.01
        min_value = np.min(array)
        if min_value < 0.01:
            array += (0.01 - min_value)
            array *= total_sum / np.sum(array)  # Renormalize after adjustment

        # Ensure the maximum value is higher (e.g., closer to 1.0)
        max_value = np.max(array)
        if max_value < 1.0:
            # Increase the overall scaling factor
            array *= 1.0 / max_value
            # Renormalize to ensure the total sum is still 2000
            array *= total_sum / np.sum(array)

        return array


