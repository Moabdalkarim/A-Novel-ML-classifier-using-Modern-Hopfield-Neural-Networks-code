# Imports
import cupy as cp  
from scipy.special import logsumexp  
from sklearn.decomposition import PCA  
import time
from functools import reduce
import numpy as np

# functions
def is_gpu_available_cupy():
    """
    Check if a GPU is available using CuPy.

    Returns:
        bool: True if a GPU is available, False otherwise.
    """
    try:
        # Attempt to create a small CuPy array and allocate it on the GPU
        _ = cp.array([1])
        return True
    except cp.cuda.runtime.CUDARuntimeError:
        return False


# check if GPU avaible to GPU code, if not use CPU code
if is_gpu_available_cupy():
    print("GPU is available, using Hopfield_Classifier-GPU version")
    GPU_flg=True
else:
    print("GPU is not available, using Hopfield_Classifier-CPU version")
    GPU_flg=False


# Hopfield_Classifier class GPU version
if GPU_flg:
  class Hopfield_Classifier:
      def __init__(self,w_compress=0, PCA=0):
          """
          Initialize the Hopfield Network.
          """
          # Initializing variables for network memory, weights, state, and to store energies
          self.memory = None
          self.weights = None
          self.pca_model = None
          self.energies = []
          self.w_compress= w_compress
          self.PCA= PCA
          

      @staticmethod
      def shuffle_along_axis(a, axis):
          """
          Shuffle the elements of an array along a specified axis.

          Args:
          a (array): The array to shuffle.
          axis (int): The axis along which to shuffle the array.

          Returns:
          array: The shuffled array.
          """
          # Generating indices to shuffle along the specified axis
          idx = cp.random.rand(*a.shape).argsort(axis=axis)
          # Returning the shuffled array
          return cp.take_along_axis(a, idx, axis=axis)

      @staticmethod
      def calc_PCA(class_data, n_components):
          """
          Calculate PCA (Principal Component Analysis) for the given data.

          Args:
          class_data (array): The data for PCA calculation.
          n_components (int/float): Number of principal components to consider/explained variance percentage.

          Returns:
          tuple: Transformed data, PCA model, and explained variance.
          """
          # Initializing PCA with specified number of components
          pca = PCA(n_components=n_components)
          # Transforming the data using PCA
          transformed_data = pca.fit_transform(class_data.get())
          # Calculating explained variance
          explained_variance = cp.sum(pca.explained_variance_ratio_)
          # Returning transformed data, the PCA model, and explained variance
          return transformed_data, pca, explained_variance

      @staticmethod
      def dataset_hnn_preprocess(x_train, y_train, silent=True):
          """
          Preprocess the dataset for Hopfield Neural Network (HNN) training by balancing classes.

          Args:
          x_train (array): Training data features.
          y_train (array): Training data labels.
          PCA (int/float): Number of PCA components to apply/explained variance percentage (default 0, no PCA).
          silent (bool): If True, suppresses print statements.

          Returns:
          tuple: List of classes data.
          """
          # Finding unique targets (classes) in the training labels
          targets = cp.unique(y_train)
          # Counting the number of samples for each class
          classes_numbers_list = [cp.argwhere(y_train == target).shape[0] for target in targets]
          # Finding the maximum number of samples in any class
          max_num = cp.array(classes_numbers_list).max()

          # List to hold processed data for each class
          classes_list = []
          for target in targets:
              # Finding indices of the current target class
              idx = cp.argwhere(y_train == target)
              # Extracting features for the current target class
              x_trgt = x_train[idx[:, 0]]
              # Optional print statements for data stats
              if not silent:
                  print(f"class {target} : {x_trgt.shape} ==> min: {x_trgt.min()}, mean: {x_trgt.mean()}, max: {x_trgt.max()}")
              # Balancing classes by repeating samples if necessary
              x_trgt = cp.concatenate((x_trgt, x_trgt[:max_num - len(x_trgt)]), axis=0)
              # Appending processed data to the list
              classes_list.append(x_trgt)

          # Returning the list of processed class data
          return classes_list

      def fit(self, x_train, y_train, silent=True):
          """
          Train Hopfield Neural Networks for each class in the dataset.

          Args:
          x_train (array): Training data features.
          y_train (array): Training data labels.
          w_compress (float): Weight compression factor (default 0, no compression).
          PCA (int/float): Number of PCA components to apply/explained variance percentage (default 0, no PCA).
          silent (bool): If True, suppresses print statements.

          Returns:
          object: Trained Hopfield Network model.
          """

          # reshape and convert to cupy array
          x_train = x_train.reshape(-1, reduce(lambda x, y: x * y, (x_train.shape[i] for i in range(1, len(x_train.shape)))))
          y_train=y_train.reshape((-1))
          x_train = cp.array(x_train)
          y_train = cp.array(y_train)

          # Recording start time for training
          start_time = time.time()
          # Preprocessing the dataset
          classes_list = self.dataset_hnn_preprocess(x_train, y_train, silent)

          # Initializing list to store memory for each class
          memories_list = cp.array(classes_list)

          # Weight compression, if enabled
          if self.w_compress != 0:
              if not silent:
                  # Optional print statements for weight compression
                  print("Weight compression enabled by value: ", self.w_compress)
                  print("H_Net_trgt shape before: ", memories_list.shape)
              # Shuffling along the second axis
              self.shuffle_along_axis(memories_list, axis=1)
              # Reducing the number of samples based on compression factor
              memories_list = memories_list[:, 0:int(memories_list.shape[1] * self.w_compress), :]
              if not silent:
                  # Optional print statements after compression
                  print("H_Net_trgt shape after: ", memories_list.shape)

          # PCA compression, if enabled
          if self.PCA != 0:
              # Reshaping data for PCA
              reshaped_memories_list = memories_list.reshape((memories_list.shape[0] * memories_list.shape[1], memories_list.shape[2]))
              if not silent:
                  # Optional print statements for PCA
                  print("PCA compression enabled by value: ", self.PCA)
                  print("H_Net_trgt shape before: ", memories_list.shape)
              # Applying PCA for compression
              trans_memories_list, pca_model, explained_var = self.calc_PCA(reshaped_memories_list, self.PCA)
              # Reshaping back after PCA
              memories_list = trans_memories_list.reshape((memories_list.shape[0], memories_list.shape[1], -1))
              if not silent:
                  # Optional print statements after PCA
                  print("Explained variance : ", explained_var)
                  print("H_Net_trgt shape after: ", memories_list.shape)

          # Storing the processed memories in the network
          self.memory = cp.array(memories_list)
          # Learning the weights based on the memories
          self.network_learning()
          # Storing the PCA model if used
          if self.PCA != 0:
              self.pca_model = pca_model
          if not silent:
              # Printing the total training time
              print("\nTraining time : ", time.time() - start_time)

      def network_learning(self):
          """
          Method for the network to learn or store the patterns.
          """
          # Using the memories directly as weights
          X = self.memory
          self.weights = X

      def predict(self, x_test, patch_size=-1, silent=True):
          """
          Predict the class of multiple input instances using Hopfield Neural Network.

          Args:
          x_test (array): Test data features.
          patch_size (int): Patch size for energy computation (default -1, full length).
          PCA (bool): If True, apply PCA transformation.
          silent (bool): If True, suppresses print statements.

          Returns:
          array: Predicted labels for the test data.
          """

          # reshape and convert to cupy array
          x_test = x_test.reshape(-1, reduce(lambda x, y: x * y, (x_test.shape[i] for i in range(1, len(x_test.shape)))))
          x_test = cp.array(x_test)

          # Recording the start time for prediction
          start_time = time.time()
          if self.pca_model != None:
              # Start time for PCA transformation
              pca_start_time = time.time()
              # Applying PCA transformation if enabled
              x_test = cp.array(self.pca_model.transform(x_test.get()))
              # End time for PCA transformation
              pca_end_time = time.time()
              if not silent:
                  # Printing PCA transformation time
                  print("PCA transform only time: ", pca_end_time - pca_start_time)

          # Start time for energy computation
          energy_start_time = time.time()
          # Computing energy for the test data
          self.compute_energy_ext(x_test, patch_size)
          # Storing the predicted labels
          y_pred = self.mem_ext
          # End time for energy computation
          energy_end_time = time.time()
          if not silent:
              # Printing inference and energy computation time
              print("Inference time : ", time.time() - start_time)
              print("Energy computation time: ", energy_end_time - energy_start_time)

          # Returning the predicted labels
          return(np.array(cp.array(y_pred).get()))

      def compute_energy_ext(self, input_state, patch_size=-1):
          """
          Compute the energy of the network for an external input state.

          Args:
          input_state (array): The state for which the energy is to be computed.
          patch_size (int, optional): Size of the patch for energy computation. Defaults to -1, meaning the entire length.
          """
          # Transposing the weights for computation
          ww = cp.transpose(self.weights, axes=(0, 2, 1))
          # Defaulting patch size to the length of input if not specified
          if patch_size == -1:
              patch_size = len(input_state)

          # List to store results for each patch
          result_list = []
          for i in range(0, len(input_state) // patch_size + 1):
              # Extracting a chunk of the input state
              A_chunk = input_state[i * patch_size:(i * patch_size + patch_size)]
              # Computing the result for the chunk
              result_chunk = cp.dot(A_chunk, ww).get()
              # Calculating the log sum exponent for stability
              logsum = -(logsumexp(result_chunk.T, axis=0))
              # Finding the argmin of the logsum
              result_chunk_argmin = cp.argmin(cp.array(logsum), axis=0)
              # Appending the result to the list
              result_list.append(result_chunk_argmin)

          # Concatenating the results to form the extended memory
          self.mem_ext = cp.concatenate(tuple(result_list), axis=0)


# Hopfield_Classifier class CPU version
else :
  class Hopfield_Classifier:
      def __init__(self,w_compress=0, PCA=0):
          """
          Initialize the Hopfield Network.
          """
          # Initializing variables for network memory, weights, state, and to store energies
          self.memory = None
          self.weights = None
          self.pca_model = None
          self.energies = []
          self.w_compress= w_compress
          self.PCA= PCA
          

      @staticmethod
      def shuffle_along_axis(a, axis):
          """
          Shuffle the elements of an array along a specified axis.

          Args:
          a (array): The array to shuffle.
          axis (int): The axis along which to shuffle the array.

          Returns:
          array: The shuffled array.
          """
          # Generating indices to shuffle along the specified axis
          idx = np.random.rand(*a.shape).argsort(axis=axis)
          # Returning the shuffled array
          return np.take_along_axis(a, idx, axis=axis)

      @staticmethod
      def calc_PCA(class_data, n_components):
          """
          Calculate PCA (Principal Component Analysis) for the given data.

          Args:
          class_data (array): The data for PCA calculation.
          n_components (int/float): Number of principal components to consider/explained variance percentage.

          Returns:
          tuple: Transformed data, PCA model, and explained variance.
          """
          # Initializing PCA with specified number of components
          pca = PCA(n_components=n_components)
          # Transforming the data using PCA
          transformed_data = pca.fit_transform(class_data)
          # Calculating explained variance
          explained_variance = np.sum(pca.explained_variance_ratio_)
          # Returning transformed data, the PCA model, and explained variance
          return transformed_data, pca, explained_variance

      @staticmethod
      def dataset_hnn_preprocess(x_train, y_train, silent=True):
          """
          Preprocess the dataset for Hopfield Neural Network (HNN) training by balancing classes.

          Args:
          x_train (array): Training data features.
          y_train (array): Training data labels.
          PCA (int/float): Number of PCA components to apply/explained variance percentage (default 0, no PCA).
          silent (bool): If True, suppresses print statements.

          Returns:
          tuple: List of classes data.
          """
          # Finding unique targets (classes) in the training labels
          targets = np.unique(y_train)
          # Counting the number of samples for each class
          classes_numbers_list = [np.argwhere(y_train == target).shape[0] for target in targets]
          # Finding the maximum number of samples in any class
          max_num = np.array(classes_numbers_list).max()

          # List to hold processed data for each class
          classes_list = []
          for target in targets:
              # Finding indices of the current target class
              idx = np.argwhere(y_train == target)
              # Extracting features for the current target class
              x_trgt = x_train[idx[:, 0]]
              # Optional print statements for data stats
              if not silent:
                  print(f"class {target} : {x_trgt.shape} ==> min: {x_trgt.min()}, mean: {x_trgt.mean()}, max: {x_trgt.max()}")
              # Balancing classes by repeating samples if necessary
              x_trgt = np.concatenate((x_trgt, x_trgt[:max_num - len(x_trgt)]), axis=0)
              # Appending processed data to the list
              classes_list.append(x_trgt)

          # Returning the list of processed class data
          return classes_list

      def fit(self, x_train, y_train, silent=True):
          """
          Train Hopfield Neural Networks for each class in the dataset.

          Args:
          x_train (array): Training data features.
          y_train (array): Training data labels.
          w_compress (float): Weight compression factor (default 0, no compression).
          PCA (int/float): Number of PCA components to apply/explained variance percentage (default 0, no PCA).
          silent (bool): If True, suppresses print statements.

          Returns:
          object: Trained Hopfield Network model.
          """

          # reshape and convert to cupy array
          x_test = x_test.reshape(-1, reduce(lambda x, y: x * y, (x_test.shape[i] for i in range(1, len(x_test.shape)))))
          y_train=y_train.reshape((-1))
          x_train = np.array(x_train)
          y_train = np.array(y_train)

          # Recording start time for training
          start_time = time.time()
          # Preprocessing the dataset
          classes_list = self.dataset_hnn_preprocess(x_train, y_train, silent)

          # Initializing list to store memory for each class
          memories_list = np.array(classes_list)

          # Weight compression, if enabled
          if self.w_compress != 0:
              if not silent:
                  # Optional print statements for weight compression
                  print("Weight compression enabled by value: ", self.w_compress)
                  print("H_Net_trgt shape before: ", memories_list.shape)
              # Shuffling along the second axis
              self.shuffle_along_axis(memories_list, axis=1)
              # Reducing the number of samples based on compression factor
              memories_list = memories_list[:, 0:int(memories_list.shape[1] * self.w_compress), :]
              if not silent:
                  # Optional print statements after compression
                  print("H_Net_trgt shape after: ", memories_list.shape)

          # PCA compression, if enabled
          if self.PCA != 0:
              # Reshaping data for PCA
              reshaped_memories_list = memories_list.reshape((memories_list.shape[0] * memories_list.shape[1], memories_list.shape[2]))
              if not silent:
                  # Optional print statements for PCA
                  print("PCA compression enabled by value: ", self.PCA)
                  print("H_Net_trgt shape before: ", memories_list.shape)
              # Applying PCA for compression
              trans_memories_list, pca_model, explained_var = self.calc_PCA(reshaped_memories_list, self.PCA)
              # Reshaping back after PCA
              memories_list = trans_memories_list.reshape((memories_list.shape[0], memories_list.shape[1], -1))
              if not silent:
                  # Optional print statements after PCA
                  print("Explained variance : ", explained_var)
                  print("H_Net_trgt shape after: ", memories_list.shape)

          # Storing the processed memories in the network
          self.memory = np.array(memories_list)
          # Learning the weights based on the memories
          self.network_learning()
          # Storing the PCA model if used
          if self.PCA != 0:
              self.pca_model = pca_model
          if not silent:
              # Printing the total training time
              print("\nTraining time : ", time.time() - start_time)

      def network_learning(self):
          """
          Method for the network to learn or store the patterns.
          """
          # Using the memories directly as weights
          X = self.memory
          self.weights = X

      def predict(self, x_test, patch_size=-1, silent=True):
          """
          Predict the class of multiple input instances using Hopfield Neural Network.

          Args:
          x_test (array): Test data features.
          patch_size (int): Patch size for energy computation (default -1, full length).
          PCA (bool): If True, apply PCA transformation.
          silent (bool): If True, suppresses print statements.

          Returns:
          array: Predicted labels for the test data.
          """

          # reshape and convert to cupy array
          x_test = x_test.reshape(-1, reduce(lambda x, y: x * y, (x_train.shape[i] for i in range(1, len(x_train.shape)))))
          x_test = np.array(x_test)

          # Recording the start time for prediction
          start_time = time.time()
          if self.pca_model != None:
              # Start time for PCA transformation
              pca_start_time = time.time()
              # Applying PCA transformation if enabled
              x_test = np.array(self.pca_model.transform(x_test))
              # End time for PCA transformation
              pca_end_time = time.time()
              if not silent:
                  # Printing PCA transformation time
                  print("PCA transform only time: ", pca_end_time - pca_start_time)

          # Start time for energy computation
          energy_start_time = time.time()
          # Computing energy for the test data
          self.compute_energy_ext(x_test, patch_size)
          # Storing the predicted labels
          y_pred = self.mem_ext
          # End time for energy computation
          energy_end_time = time.time()
          if not silent:
              # Printing inference and energy computation time
              print("Inference time : ", time.time() - start_time)
              print("Energy computation time: ", energy_end_time - energy_start_time)

          # Returning the predicted labels
          return(np.array(y_pred))

      def compute_energy_ext(self, input_state, patch_size=-1):
          """
          Compute the energy of the network for an external input state.

          Args:
          input_state (array): The state for which the energy is to be computed.
          patch_size (int, optional): Size of the patch for energy computation. Defaults to -1, meaning the entire length.
          """
          # Transposing the weights for computation
          ww = np.transpose(self.weights, axes=(0, 2, 1))
          # Defaulting patch size to the length of input if not specified
          if patch_size == -1:
              patch_size = len(input_state)

          # List to store results for each patch
          result_list = []
          for i in range(0, len(input_state) // patch_size + 1):
              # Extracting a chunk of the input state
              A_chunk = input_state[i * patch_size:(i * patch_size + patch_size)]
              # Computing the result for the chunk
              result_chunk = np.dot(A_chunk, ww)
              # Calculating the log sum exponent for stability
              logsum = -(logsumexp(result_chunk.T, axis=0))
              # Finding the argmin of the logsum
              result_chunk_argmin = np.argmin(np.array(logsum), axis=0)
              # Appending the result to the list
              result_list.append(result_chunk_argmin)

          # Concatenating the results to form the extended memory
          self.mem_ext = np.concatenate(tuple(result_list), axis=0)
