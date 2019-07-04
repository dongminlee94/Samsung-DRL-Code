######################
##### 1. Ndarray #####
######################

## 1.1 Numpy import
import numpy as np

##################################################################

## 1.2 Array creation
array_creation = np.array([[1, 4, 5, "8"], [3, 5, 6, 7]], float)
# print(array_creation)
# print(type(array_creation))
# print(array_creation.shape)
# print(array_creation.dtype)
'''
[[1. 4. 5. 8.]
 [3. 5. 6. 7.]]
<class 'numpy.ndarray'>
(2, 4)
float64
'''

### 1.2.1 Array shape

#### Vector
vector = [1,2,3,4]
# print(np.array(vector, float).shape)
'''
(4,)
'''

#### Matrix
matrix = [[1,2,5,8], [1,2,5,8], [1,2,5,8]]
# print(np.array(matrix, float).shape)
'''
(3, 4)
'''

#### Tensor
tensor = [[[1,2,5,8], [1,2,5,8], [1,2,5,8]], 
          [[1,2,5,8], [1,2,5,8], [1,2,5,8]], 
          [[1,2,5,8], [1,2,5,8], [1,2,5,8]], 
          [[1,2,5,8], [1,2,5,8], [1,2,5,8]]]
# print(np.array(tensor, float).shape)
'''
(4, 3, 4)
'''

### 1.2.2 Array dtype
array_dtype = np.array([[1, 2, 3], [4.5, 5, 6]], dtype=int)
# print(array_dtype)
'''
[[1 2 3]
 [4 5 6]]
'''

array_dtype1 = np.array([[1, 2, 3], [4.5, 5, 6]])
# print(array_dtype1)
'''
[[1.  2.  3. ]
 [4.5 5.  6. ]]
'''

##################################################################
##################################################################

##########################################
##### 2. Reshape & Indexing, Slicing #####
##########################################

## 2.1 Reshape

matrix = [[1,2,3,4], [1,2,5,8]]

reshape_array = np.array(matrix).reshape(8)
# print(reshape_array)
'''
[1 2 3 4 1 2 5 8]
'''

reshape_array1 = np.reshape(matrix, [1, 8])
# print(reshape_array1)
'''
[[1 2 3 4 1 2 5 8]]
'''

##################################################################

## 2.2 Indexing
array_indexing = np.array([[1, 2, 3], [4, 5, 6]])
# print(array_indexing[0][0])
# print(array_indexing[0,0])
'''
1
1
'''

array_indexing[0,0] = 10 # matrix 0,0에 10 할당
# print(array_indexing)
'''
[[10  2  3]
 [ 4  5  6]]
'''

##################################################################

## 2.3 Slicing
array_slicing = np.array([[1,2,5,8], [1,3,6,9], [1,4,7,10], [1,2,3,4]])
# print(array_slicing[:2, :])
# print(array_slicing[:, 1:3])
'''
[[1 2 5 8]
 [1 3 6 9]]
[[2 5]
 [3 6]
 [4 7]
 [2 3]]
'''

##################################################################
##################################################################

#################################
##### 3. Creating Functions #####
#################################

## 3.1 Zeros & Ones
zeros = np.zeros(shape=(10))
zeros1 = np.zeros((2, 5))
# print(zeros)
# print(zeros1)
'''
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
[[0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]]
'''

ones = np.ones(10)
ones1 = np.ones((5, 2))
# print(ones)
# print(ones1)
'''
[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
[[1. 1.]
 [1. 1.]
 [1. 1.]
 [1. 1.]
 [1. 1.]]
'''

##################################################################

## 3.2 Something_like
matrix = np.array([[1,3,5,7], [2,4,8,10]])
zeros = np.zeros_like(matrix)
# print(zeros)
'''
[[0 0 0 0]
 [0 0 0 0]]
'''

##################################################################

## 3.3 Random sampling

### Uniform distribution
uniform_sampling = np.random.uniform(0, 1, (2,5))
# print(uniform_sampling)
'''
[[0.26639967 0.62071191 0.67649403 0.45036727 0.01181673]
 [0.52986859 0.84493667 0.47515631 0.28743741 0.77223663]]
'''

### Uniform distribution over [0, 1)
rand_sampling = np.random.rand(2,5)
# print(rand_sampling)
'''
[[0.38000795 0.40664224 0.51519862 0.21076549 0.8430714 ]
 [0.67438271 0.16272181 0.1998186  0.9505862  0.76784629]]
'''

### Normal(Gaussian) distribution
normal_sampling = np.random.normal(0, 1, (2,5))
# print(normal_sampling)
'''
[[-0.6013544   1.89544317  1.45838014 -0.48518027  1.11360633]
 [ 1.1276201  -0.99321183 -0.37415802 -2.25353429 -1.09036642]]
'''

### Standard normal(gaussian) distribution of mean 0 and variance 1 
randn_sampling = np.random.randn(2,5)
# print(randn_sampling)
'''
[[ 0.92879391 -0.56569603 -0.30848259 -1.42508296 -0.92830748]
 [ 0.45703894 -0.24427407  0.45944466 -0.48407352 -0.3964146 ]]
'''

##################################################################
##################################################################

##################################
##### 4. Operation Functions #####
##################################

## 4.1 Sum
matrix = np.array([[1,3,5,7], [2,4,8,10]])
matrix_sum = matrix.sum()
# print(matrix_sum)
'''
40
'''

##################################################################

## 4.2 Axis

### Matrix
matrix = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

matrix_axis_0 = matrix.sum(axis=0)
# print(matrix_axis_0)
'''
[15 18 21 24]
'''

matrix_axis_1 = matrix.sum(axis=1)
# print(matrix_axis_1)
'''
[10 26 42]
'''

matrix_axis_2 = matrix.sum(axis=-1)
# print(matrix_axis_2)
'''
[10 26 42]
'''

### Tensor
tensor = np.array([matrix, matrix, matrix])
# print(third_order_tensor)
# print(third_order_tensor.shape)
'''
[[[ 1  2  3  4]
  [ 5  6  7  8]
  [ 9 10 11 12]]

 [[ 1  2  3  4]
  [ 5  6  7  8]
  [ 9 10 11 12]]

 [[ 1  2  3  4]
  [ 5  6  7  8]
  [ 9 10 11 12]]]
(3, 3, 4)
'''

tensor_axis_0 = tensor.sum(axis=0)
# print(tensor_axis_0)
'''
[[ 3  6  9 12]
 [15 18 21 24]
 [27 30 33 36]]
'''

tensor_axis_1 = tensor.sum(axis=1)
# print(tensor_axis_1)
'''
[[15 18 21 24]
 [15 18 21 24]
 [15 18 21 24]]
'''

tensor_axis_2 = tensor.sum(axis=2)
# print(tensor_axis_2)
'''
[[10 26 42]
 [10 26 42]
 [10 26 42]]
'''

tensor_axis_3 = tensor.sum(axis=-1)
# print(tensor_axis_3)
'''
[[10 26 42]
 [10 26 42]
 [10 26 42]]
'''

##################################################################

## 4.3 Mean & Std
matrix = np.array([[1,2,3,4], [5,6,7,8]])
mean = np.mean(matrix)
std = np.std(matrix)
# print(mean)
# print(std)
'''
4.5
2.29128784747792
'''

##################################################################

## 4.4 Mathematical functions
matrix = np.array([[1,2,3,4], [5,6,7,8]])
exp = np.exp(matrix)
sqrt = np.sqrt(matrix)
# print(exp)
# print(sqrt)
'''
[[2.71828183e+00 7.38905610e+00 2.00855369e+01 5.45981500e+01]
 [1.48413159e+02 4.03428793e+02 1.09663316e+03 2.98095799e+03]]
[[1.         1.41421356 1.73205081 2.        ]
 [2.23606798 2.44948974 2.64575131 2.82842712]]
'''

##################################################################

## 4.5 Concatenate

### Vstack
a = np.array([1, 2, 3])
b = np.array([2, 3, 4])
vstack = np.vstack((a,b))
# print(vstack)
'''
[[1 2 3]
 [2 3 4]]
'''

### Hstack
a = np.array([[1], [2], [3]])
b = np.array([[2], [3], [4]])
hstack = np.hstack((a,b))
# print(hstack)
'''
[[1 2]
 [2 3]
 [3 4]]
'''

### Concatenate
a = np.array([[1, 2, 3]])
b = np.array([[2, 3, 4]])

concat = np.concatenate((a,b)) # vstack
# print(concat)
'''
[[1 2 3]
 [2 3 4]]
'''

concat_0 = np.concatenate((a,b), axis=0) # vstack
# print(concat_0)
'''
[[1 2 3]
 [2 3 4]]
'''

concat_1 = np.concatenate((a,b), axis=1) # hstack
# print(concat_1)
'''
[[1 2 3 2 3 4]]
'''

concat_none = np.concatenate((a,b), axis=None) # flatten
# print(concat_none)
'''
[1 2 3 2 3 4]
'''

##################################################################

## 4.6 Element-wise operations
matrix = np.array([[1,2,3,4], [5,6,7,8]])
element_wise = matrix * matrix
# print(element_wise)
'''
[[ 1  4  9 16]
 [25 36 49 64]]
'''

##################################################################

## 4.7 Dot product
matrix = np.array([[1,2,3,4], [5,6,7,8]])

matrix_a = np.reshape(matrix, (4, 2))
# print(matrix_a)
'''
[[1 2]
 [3 4]
 [5 6]
 [7 8]]
'''

dot = np.dot(matrix, matrix_a)
# print(dot)
'''
[[ 50  60]
 [114 140]]
'''

##################################################################

## 4.8 Transpose
matrix = np.array([[1,2,3,4], [5,6,7,8]])

transpose = np.transpose(matrix)
# print(transpose)
'''
[[1 5]
 [2 6]
 [3 7]
 [4 8]]
'''

transpose1 = matrix.transpose()
transpose2 = matrix.T
# print(transpose1)
# print(transpose2)
'''
[[1 5]
 [2 6]
 [3 7]
 [4 8]]
[[1 5]
 [2 6]
 [3 7]
 [4 8]]
'''

##################################################################

## 4.9 Broadcasting

matrix = np.array([[1,2,3], [4,5,6]])
scalar = 3

# print(matrix + scalar)        # add
'''
[[4 5 6]
 [7 8 9]]
'''

# print(matrix - scalar)        # subtract
'''
[[-2 -1  0]
 [ 1  2  3]]
'''

# print(matrix * scalar)        # multiply 
'''
[[ 3  6  9]
 [12 15 18]]
'''

# print(matrix / scalar)        # divide 
'''
[[0.33333333 0.66666667 1.        ]
 [1.33333333 1.66666667 2.        ]]
'''

# print(matrix // scalar)       # quotient
'''
[[0 0 1]
 [1 1 2]]
'''

# print(matrix % scalar)        # remainder
'''
[[1 2 0]
 [1 2 0]]
'''

# print(matrix ** scalar)       # power
'''
[[  1   8  27]
 [ 64 125 216]]
'''
