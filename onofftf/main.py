import os
import numpy as np
import tensorflow as tf
import gpflow as gp
from gpflow import transforms
from tensorflow.python.framework import random_seed
import numpy
import matplotlib.pyplot as plt
from tensorflow import Variable
float_type = tf.float64
jitter_level = 1e-8
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Variable(Variable):
    '''
    extend tf.Variable to have properties : learning_rate
    '''
    pass

    def set_learning_rate(self,value):
        self._learning_rate = value

    @property
    def learning_rate(self):
        if hasattr(self,'_learning_rate'):
            return self._learning_rate

        else:
            return 0.001


class KernSE:
    '''
    Copied and modified from GPflow(https://github.com/GPflow/)
    
    Implementation of Squared exponential kernels 

    Input:
	X           - Tensorflow placeholder for data
	Lengthscale - instances of Param object
	Variance    - instances of Param object
   '''
    def __init__(self, lengthscales,variance):
        self.lengthscales = lengthscales.get_tfv()
        self.variance = variance.get_tfv()

    def square_dist(self,X, X2=None):
        X = X / self.lengthscales
        Xs = tf.reduce_sum(tf.square(X), 1)
        if X2 is None:
            return -2 * tf.matmul(X, X, transpose_b=True) + \
                   tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
        else:
            X2 = X2 / self.lengthscales
            X2s = tf.reduce_sum(tf.square(X2), 1)
            return -2 * tf.matmul(X, X2, transpose_b=True) + \
                   tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))

    def K(self,X,X2=None):
        if X2 is None:
            return self.variance * tf.exp(-self.square_dist(X) / 2)
        else:
            return self.variance * tf.exp(-self.square_dist(X, X2) / 2)

    def Ksymm(self,X):
        return self.variance * tf.exp(-self.square_dist(X) / 2)

    def Kdiag(self,X):
        return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))


class DataSet(object):
    def __init__(self,
               xtrain,
               ytrain,
               dtype=float_type,
               seed=121):
        seed1, seed2 = random_seed.get_seed(seed)
        numpy.random.seed(seed1 if seed is None else seed2)
        self._num_examples = xtrain.shape[0]

        self._xtrain = xtrain
        self._ytrain = ytrain
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def xtrain(self):
        return self._xtrain

    @property
    def ytrain(self):
        return self._ytrain

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch

        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:

            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
            self._xtrain = self.xtrain[perm0]
            self._ytrain = self.ytrain[perm0]

        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            xtrain_rest_part = self._xtrain[start:self._num_examples]
            ytrain_rest_part = self._ytrain[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._xtrain = self.xtrain[perm]
                self._ytrain = self.ytrain[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            xtrain_new_part = self._xtrain[start:end]
            ytrain_new_part = self._ytrain[start:end]
            return numpy.concatenate((xtrain_rest_part, xtrain_new_part), axis=0) , numpy.concatenate((ytrain_rest_part, ytrain_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._xtrain[start:end], self._ytrain[start:end]



class Param:
    '''
    Copied and modified from GPflow(https://github.com/GPflow/)
    '''
    def __init__(self,value,transform = None,fixed=False,name=None,learning_rate=None,summ=False):
        self.value = value
        self.fixed = fixed

        if name is None:
            self.name = "param"
        else:
            self.name = name

        if transform is None:
            self.transform=transforms.Identity()
        else:
            self.transform = transform

        if self.fixed:
            self.tf_opt_var = tf.constant(self.value,name=name,dtype=float_type)
        else:
            self.tf_opt_var = Variable(self.transform.backward(self.value),name=name,dtype=float_type)

        if learning_rate is not None and fixed is False:
            self.tf_opt_var.set_learning_rate(learning_rate)

        if summ:
            self.variable_summaries(self.tf_opt_var)

    def get_optv(self):
        return self.tf_opt_var

    def get_tfv(self):
        if self.fixed:
            return self.tf_opt_var
        else:
            return self.transform.tf_forward(self.tf_opt_var)

    def variable_summaries(self,var):
      tf.summary.histogram(self.name, var)

    @property
    def shape(self):
        return self.value.shape


def GaussKL(q_mu, q_sqrt, K=None):
    """
    Copied and modified from GPflow(https://github.com/GPflow/)
    TODO: remove num_func computations : make return shapes N x 1 and N x N
    """

    if K is None:
        white = True
        alpha = q_mu
    else:
        white = False
        num_data = tf.shape(K)[0]
        K = K + tf.eye(num_data, dtype=float_type) * jitter_level
        Lp = tf.cholesky(K)
        alpha = tf.matrix_triangular_solve(Lp, q_mu, lower=True)

    if q_sqrt.get_shape().ndims == 2:
        diag = True
        num_latent = tf.shape(q_sqrt)[1]
        NM = tf.size(q_sqrt)
        Lq = Lq_diag = q_sqrt
    elif q_sqrt.get_shape().ndims == 3:
        diag = False
        num_latent = tf.shape(q_sqrt)[2]
        NM = tf.reduce_prod(tf.shape(q_sqrt)[1:])
        Lq = tf.matrix_band_part(tf.transpose(q_sqrt, (2, 0, 1)), -1, 0)  # force lower triangle
        Lq_diag = tf.matrix_diag_part(Lq)
    else: 
        raise ValueError("Bad dimension for q_sqrt: {}".format(q_sqrt.get_shape().ndims))

    # Mahalanobis term: μqᵀ Σp⁻¹ μq
    mahalanobis = tf.reduce_sum(tf.square(alpha))

    # Constant term: - N x M
    constant = - tf.cast(NM, float_type)

    # Log-determinant of the covariance of q(x):
    logdet_qcov = tf.reduce_sum(tf.log(tf.square(Lq_diag)))

    # Trace term: tr(Σp⁻¹ Σq)
    if white:
        trace = tf.reduce_sum(tf.square(Lq))
    else:
        if diag:
            M = tf.shape(Lp)[0]
            Lp_inv = tf.matrix_triangular_solve(
                Lp, tf.eye(M, dtype=float_type), lower=True)
            K_inv = tf.matrix_triangular_solve(
                tf.transpose(Lp), Lp_inv, lower=False)
            trace = tf.reduce_sum(
                tf.expand_dims(tf.matrix_diag_part(K_inv), 1) * tf.square(q_sqrt))
        else:
            Lp_tiled = tf.tile(tf.expand_dims(Lp, 0), [num_latent, 1, 1])
            LpiLq = tf.matrix_triangular_solve(Lp_tiled, Lq, lower=True)
            trace = tf.reduce_sum(tf.square(LpiLq))

    twoKL = mahalanobis + constant - logdet_qcov + trace

    # Log-determinant of the covariance of p(x):
    if not white:
        log_sqdiag_Lp = tf.log(tf.square(tf.matrix_diag_part(Lp)))
        sum_log_sqdiag_Lp = tf.reduce_sum(log_sqdiag_Lp)
        prior_logdet = tf.cast(num_latent, float_type) * sum_log_sqdiag_Lp
        twoKL += prior_logdet

    return 0.5 * twoKL


def GPConditional(Xnew, X, kern, f, full_cov=False, q_sqrt=None,whiten=False):
    """
    Copied and modified from GPflow(https://github.com/GPflow/)

    TODO: remove num_func computations : make return shapes N x 1 and N x N
    """

    # compute kernel stuff
    num_data = tf.shape(X)[0]  # M
    num_func = tf.shape(f)[1]  # K
    Kmn = kern.K(X, Xnew)
    Kmm = kern.K(X) + tf.eye(num_data, dtype=float_type) * jitter_level
    Lm = tf.cholesky(Kmm)

    # Compute the projection matrix A
    A = tf.matrix_triangular_solve(Lm, Kmn, lower=True)

    # compute the covariance due to the conditioning
    if full_cov:
        fvar = kern.K(Xnew) - tf.matmul(A, A, transpose_a=True)
        shape = tf.stack([num_func, 1, 1])
    else:
        fvar = kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
        shape = tf.stack([num_func, 1])
    fvar = tf.tile(tf.expand_dims(fvar, 0), shape)  # K x N x N or K x N

    # another backsubstitution in the unwhitened case
    if not whiten:
        A = tf.matrix_triangular_solve(tf.transpose(Lm), A, lower=False)

    # construct the conditional mean
    fmean = tf.matmul(A, f, transpose_a=True)

    if q_sqrt is not None:
        if q_sqrt.get_shape().ndims == 2:
            LTA = A * tf.expand_dims(tf.transpose(q_sqrt), 2)  # K x M x N
        elif q_sqrt.get_shape().ndims == 3:
            L = tf.matrix_band_part(tf.transpose(q_sqrt, (2, 0, 1)), -1, 0)  # K x M x M
            A_tiled = tf.tile(tf.expand_dims(A, 0), tf.stack([1, 1, 1]))
            LTA = tf.matmul(L, A_tiled, transpose_a=True)  # K x M x N
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: %s" %
                             str(q_sqrt.get_shape().ndims))
        if full_cov:
            fvar = fvar + tf.matmul(LTA, LTA, transpose_a=True)  # K x N x N
        else:
            fvar = fvar + tf.reduce_sum(tf.square(LTA), 1)  # K x N
    fvar = tf.transpose(fvar)  # N x K or N x N x K

    return fmean, fvar
