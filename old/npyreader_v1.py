import numpy as np

class Reader:
    def __init__(self, value_path, maps_path):
        self.data = np.loadtxt(value_path)
        self.length = self.data.shape[0]
        self.maps = np.load(maps_path)

    # return tuple of array (maps, values)
    def get_batch(self, num):
        rand_list = np.random.choice(self.length, num)

        x = self.maps[rand_list]
        t = self.data[rand_list][:, 1].reshape(num, 1)
        na = np.zeros([num, 7])
        
        for i in range(10):
            na[i][int(t[i])] = 1
        
        return (x, na)