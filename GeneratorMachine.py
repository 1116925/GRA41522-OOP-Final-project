import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import layers, activations, Sequential
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import numpy as np

class VAE(tf.keras.Model):
    def __init__(self, dataset:str):
        '''
        An interdace class for the VAE generator.
        @param dataset: dataset choice, either "mnist_bw" or "mnist_color"
        '''
        super().__init__() # inherit whatever from tf.keras.Model
        if dataset == "mnist_color":
            self._iscolor = True
            self.encoder = Color_enconder()
            self.decoder = Color_decoder()
        else: # Should be the same for dummy and mnist_bw
            self._iscolor = False
            self.encoder = BW_enconder()
            self.decoder = BW_decoder()

    @tf.function
    def call(self, x):
        self._X = x
        self._z = self.encoder(x)
        self._x_hat = self.decoder(self._z)
        self.vae_loss = - (self.log_diag_mvn(x, self.decoder.mu, tf.math.log(self.decoder.std)) - self.kl_divergence(self.encoder.mu, self.encoder.log_var))
        return self.vae_loss

    @tf.function
    def train(self, x, optimizer):
        '''
        Train function to update the VAE trainable variables.

        @param x: input images.
        @param optimizer: optimizer object from tensorflow.
        @return: loss value.
        '''
        with tf.GradientTape() as tape:
            loss = self.call(x)
        gradients = tape.gradient(self.vae_loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss
    
    def kl_divergence(mu, log_var):
        return 0.5 * tf.reduce_sum(tf.square(mu) + tf.exp(log_var) - log_var - 1, axis=-1) 

    def log_diag_mvn(x, mu, log_sigma):
        sum_axes = tf.range(1, tf.rank(mu))
        k = tf.cast(tf.reduce_prod(tf.shape(mu)[1:]), x.dtype)
        logp = - 0.5 * k * tf.math.log(2*np.pi) \
            - log_sigma \
            - 0.5*tf.reduce_sum(tf.square(x - mu)/tf.math.exp(2.*log_sigma),axis=sum_axes)
        return logp
    
    def plot_grid(images,N=10,C=10,figsize=(24., 28.),name='posterior'):
        '''
        Code to plot a 10x10 grid of images,
        images have size (N, H, W, C), where N (=100) is
        number of samples, H (=28) is height, W (=28) is width,
        and C (=1 black-white,=3 color) is channels.
        '''
        fig = plt.figure(figsize=figsize)
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                        nrows_ncols=(N, C),  
                        axes_pad=0,  # pad between Axes in inch.
                        )
        for ax, im in zip(grid, images):
            # Iterating over the grid returns the Axes.
            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig('./xhat_bw_'+name+'.pdf')
        plt.close()

    def visualize_latent_space(self):
        '''
        Visualize the latent space by generating images from a grid of latent vectors.
        '''
        n = 20  # figure with 20x20 digits
        digit_size = 28
        if self._iscolor:
            digit_size = 28
            channel = 3
        else:
            digit_size = 28
            channel = 1
        # Linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
        # to produce values of the latent variables z, since the prior of the latent space is Gaussian
        grid_x = np.linspace(-4, 4, n)
        grid_y = np.linspace(-4, 4, n)[::-1]

        latent_vectors = []
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]] * 1)  # batch size of 1
                latent_vectors.append(z_sample)
        latent_vectors = np.concatenate(latent_vectors, axis=0)
        x_decoded = self.decoder(tf.convert_to_tensor(latent_vectors, dtype=tf.float32))
        if channel == 1:
            x_decoded = x_decoded.numpy().reshape((n * n, digit_size, digit_size))
        else:
            x_decoded = x_decoded.numpy().reshape((n * n, digit_size, digit_size, channel))
        self.plot_grid(x_decoded, N=n, C=channel, name='latent_space')

    def generate_from_posterior(self, num_samples=100):
        '''
        Generate images from the posterior distribution q(z|x) using random samples from the test set.
        '''
        # Randomly sample num_samples images from the test set
        idx = np.random.choice(self._X.shape[0], num_samples, replace=False)
        x_samples = tf.gather(self._X, idx)
        z_samples = self.encoder(x_samples)
        x_decoded = self.decoder(z_samples)
        if self._iscolor:
            x_decoded = x_decoded.numpy().reshape((num_samples, 28, 28, 3))
        else:
            x_decoded = x_decoded.numpy().reshape((num_samples, 28, 28))
        self.plot_grid(x_decoded, N=10, C=3 if self._iscolor else 1, name='posterior')
    
    def generate_from_prior(self, num_samples=100):
        '''
        Generate images from the prior distribution p(z) using random samples from a standard normal distribution.
        '''
        latent_dim = self.encoder.latent_dim
        z_samples = tf.random.normal((num_samples, latent_dim))
        x_decoded = self.decoder(z_samples)
        if self._iscolor:
            x_decoded = x_decoded.numpy().reshape((num_samples, 28, 28, 3))
        else:
            x_decoded = x_decoded.numpy().reshape((num_samples, 28, 28))
        self.plot_grid(x_decoded, N=10, C=3 if self._iscolor else 1, name='prior')


class bicoder(layers.Layer):
    '''
    A superclass for the enconder and decoder neural networks.
    '''
    def __init__(self, activation  = 'relu'):
        self._activation = activation
        self._out = None
        self._mu = None
        self._log_var = None
        self._std = None
    
    def color_init(self, input_shape, latent_dim, filters, kernel_size, strides):
        '''
        Initialize color dataset specific parameters.
        '''
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.filters     = filters
        self.kernel_size = kernel_size
        self.strides     = strides

    def bw_init(self, input_shape, latent_dim, units):
        '''
        Initialize black and white dataset specific parameters.
        '''
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.units = units

    def _render(self, x):
        '''
        Abstract method to be implemented in subclasses.
        '''
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    @property
    def out(self):
        return self._out 
    @out.setter
    def out(self, new_out):
        self._out = new_out
    
    @property
    def mu(self):
        return self._mu
    @mu.setter
    def mu(self, new_mu):
        self._mu = new_mu
    
    @property
    def log_var(self):
        return self._log_var
    @log_var.setter
    def log_var(self, new_log_var):
        self._log_var = new_log_var
    
    @property
    def std(self):
        return self._std
    @std.setter
    def std(self, new_std):
        self._std = new_std

    @property
    def eps(self):
        return self._eps
    @eps.setter
    def eps(self, new_eps):
        self._eps = new_eps
    
    @tf.function
    def call(self, x):
        '''
        Abstract method to be implemented in subclasses.
        '''
        raise NotImplementedError("This method should be overridden by subclasses.")
    

class enconder(bicoder):
    '''
    Class that contains encoder network behaviors.
    '''
    def __init__(self, activation='relu'):
        super().__init__(activation)

    def call(self, x):       
        self.out = self._render(x)
        self.mu  = self.out[:,:self.latent_dim]
        self.log_var = self.out[:,self.latent_dim:]
        self.std = tf.math.exp(0.5*self.log_var)
        self.eps = tf.random.normal(self.mu.shape)
        return self.mu + self.eps*self.std

    

class BW_enconder(enconder):
    def __init__(self,input_shape = (28*28,), latent_dim=20, units=400, activation='relu'):
        super().bw_init(input_shape, latent_dim, units)
        super().__init__(activation)
    
    def _render(self, x):
        encoder_mlp = Sequential(
                        [ 
                        layers.InputLayer(input_shape=(self.input_shape,)),
                        layers.Dense(self.units,activation=self._activation),
                        layers.Dense(2*self.latent_dim),
                        ]
                        )
        return encoder_mlp(x)
    
class Color_enconder(enconder):

    def __init__(self, input_shape = (28,28,3), latent_dim=50, filters=32, kernel_size=3, strides=2, activation='relu'):
        super().__init__(activation)
        super().color_init(input_shape, latent_dim, filters, kernel_size, strides)

    
    def _render(self, x):
        encoder_conv = Sequential(
                        [
                        layers.InputLayer(input_shape=(self.input_shape,)),
                        layers.Conv2D(
                          filters=self.filters,   kernel_size=self.kernel_size, strides=self.strides, activation=self._activation, padding='same'),
                        layers.Conv2D(
                          filters=2*self.filters, kernel_size=self.kernel_size, strides=self.strides, activation=self._activation, padding='same'),
                        layers.Conv2D(
                          filters=4*self.filters, kernel_size=self.kernel_size, strides=self.strides, activation=self._activation, padding='same'),
                        layers.Flatten(),
                        layers.Dense(2*self.latent_dim)
                        ]
                        )
        return encoder_conv(x)

class decoder(bicoder):
    '''
    Class that contains decoder network behaviors.
    '''
    def __init__(self, activation='relu'):
        super().__init__(activation)

    def call(self, z):       
        self._out = self._render(z)
        return self._out


class BW_decoder(decoder):
    def __init__(self,input_shape = (28*28,), latent_dim=20, units=400, activation='relu', output_dim  = 28*28):
        super().bw_init(input_shape, latent_dim, units)
        super().__init__(activation)
        self.output_dim = output_dim
    
    def _render(self, x):
        decoder_mlp = Sequential(
                                [
                                layers.InputLayer(input_shape=(self.latent_dim,)),
                                layers.Dense(self.units,activation=self._activation),
                                layers.Dense(self.output_dim),
                                ]
                                )
        return decoder_mlp(x)
    
class Color_decoder(decoder):

    def __init__(self, input_shape = (28,28,3), latent_dim=50, filters=32, kernel_size=3, strides=2, activation='relu',target_shape=(4,4,128),channel_out=3):
        super().__init__(activation)
        super().color_init(input_shape, latent_dim, filters, kernel_size, strides)
        self.target_shape=target_shape
        self.channel_out=channel_out
        self.units = np.prod(self.target_shape)
    
    def _render(self, x):
        decoder_conv = Sequential(
                                [
                                layers.InputLayer(input_shape=(self.latent_dim,)),
                                layers.Dense(units=self.units, activation=self._activation),
                                layers.Reshape(target_shape=self.target_shape),
                                layers.Conv2DTranspose(filters=self.filters*2,
                                                        kernel_size=self.kernel_size,
                                                        strides=self.strides,
                                                        activation=self._activation,
                                                        output_padding=0,
                                                        padding='same'),
                                layers.Conv2DTranspose(filters=self.filters,
                                                        kernel_size=self.kernel_size,
                                                        strides=self.strides,
                                                        activation=self._activation,
                                                        output_padding=0,
                                                        padding='same'),
                                layers.Conv2DTranspose(filters=self.channel_out,
                                                        kernel_size=self.kernel_size,
                                                        strides=self.strides,
                                                        activation=self._activation,
                                                        output_padding=0,
                                                        padding='same'),
                                ]
                                )
        return decoder_conv(x)
    
    def postprocess_BWdata(x_hat):
        '''
        Postprocess the output images from the decoder to be in [0, 255] and uint8
        '''
        return tf.clip_by_value(255*x_hat, clip_value_min=0, clip_value_max=255).numpy().astype(np.uint8)