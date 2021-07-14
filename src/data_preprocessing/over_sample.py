import math
import numpy as np

class SMOTE:
    """
    A class to implement smote oversampling algorithm to balance
    the dataset.

    Attributes:
        k: int
            Number of neighbors to consider
        distance: eucliden
            Distance metric
        dims: number of dimension in data (as we have 20 features, dims=20)
    """
    def __init__(self, distance='euclidean', dims=20, k=3):
        self.k = k
        self.dims = dims
        self.distance_measure = distance
        self.index = 0

    def populate(self, N, i, narray, min_samples, k):
        """Populate the array with synthetic samples
        Attributes
        ----------
            N: int
                Number of synthetic samples to generate
            i: int
                Unique classes
            narray: numpy.ndarray
                indexes
            min_samples: numpy.ndarray
                Samples of specific class
            k: int
                Number of neighbors
        """
        while N:
            n = np.random.randint(0, k-2)
            
            difference = min_samples[narray[n]] - min_samples[i]
            gap = np.random.uniform(0,1)

            self.synthetic_array[self.index,:] = min_samples[i] + gap * difference
            
            self.index += 1
            
            N -= 1

    def k_neighbors(self, euclidean_distance, k):
        """Returns indexes of nearest neighbour"""
        nearest_index = np.zeros((euclidean_distance.shape[0],euclidean_distance.shape[0]), dtype = np.int64)
        
        indexes = np.argsort(euclidean_distance, axis=1)
        nearest_index[:,:] = indexes
        
        return nearest_index[:, 1:k]
    
    def find_kneighbors(self,X,k):
        """Find nearest neighbors
            X: numpy.ndarray
                feature data
            k: int
                Number of neighbors

        Returns
        --------
            k neighbors
        """
        euclidean_distance = np.zeros((X.shape[0],X.shape[0]), dtype = np.float32)
        
        for i in range(len(X)):
            dif = (X - X[i])**2
            dist = np.sqrt(dif.sum(axis=1))
            euclidean_distance[i] = dist
            
        return self.k_neighbors(euclidean_distance,k)
    
    def generate_samples(self, min_samples, N, k):
        """Generate synthetic samples
        Parameters
        ----------
            N: int
                Number of synthetic data to generate
            k = int
                Number of neighbours to consider
            min_samples = numpy.ndarray
                Data of a particular class
        Returns
        -------
            synthetic data
        """
        T = min_samples.shape[0]
        N = int(N/100)
        self.synthetic_array = np.zeros((N*T,self.dims))
        if self.distance_measure == 'euclidean':
            indices = self.find_kneighbors(min_samples, k)
        for i in range(indices.shape[0]):
            self.populate(N, i, indices[i], min_samples, k)
        self.newindex = 0 
        return self.synthetic_array
            
    def fit_transform(self, X, y):
        """Returns the balanced dataset
        Parameters
        ----------
            X: numpy.ndarray
                The features of imbalanced dataset
            y: numpy.ndarray
                Imbalanced labels
        """
        # find occurence of class
        occurence_class = np.eye(int(y.max()+1), int(y.max()+1))[y].sum(axis=0)
        dominant_class = np.argmax(occurence_class)
        # find occurence of dominant class
        n_occ = int(occurence_class[dominant_class].item())
        for i in range(len(occurence_class)):
            if i != dominant_class:
                number = (n_occ - occurence_class[i]) * 100 / occurence_class[i]
                N = int(math.ceil(number / 100.0)) * 100
                candidates = X[y == i]
                xs = self.generate_samples(candidates, N, self.k)
                X = np.concatenate((X,xs))
                ys = np.ones((xs.shape[0])) * i
                y = np.concatenate((y,ys))
        return X,y
