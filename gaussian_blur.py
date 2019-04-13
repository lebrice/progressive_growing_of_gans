"""
2D Gaussian Blur Keras layer.
"""
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
import tensorflow.math as math
from typing import Tuple

import tfutil


def maximum_reasonable_std(image_resolution: int) -> float:
    kernel_size = image_resolution - 1
    std = appropriate_std(kernel_size)
    return std

def appropriate_kernel_size(std: float) -> int:
    """
    Returns the appropriate gaussian kernel size to be used for a given standard deviation.
    """
    # nearest odd number to 6*std.
    return (6 * std) * 2 // 2 + 1


def appropriate_std(kernel_size):
    std = (kernel_size-1.0) / 6.0
    return std

def get_data_format(image) -> str:
    last_dim = image.shape[-1].value if isinstance(image, tf.Tensor) else image.shape[-1]
    if last_dim in (1, 3):
        return "NHWC"
    else:
        return "NCHW"


def get_image_dims(image) -> Tuple[int, int, int]:
    data_format = get_data_format(image)
    image_height = image.shape[1 if data_format == "NHWC" else 2]
    image_width = image.shape[2 if data_format == "NHWC" else -1]
    image_channels = image.shape[-1 if data_format == "NHWC" else 1]
    return image_height, image_width, image_channels

def image_at_scale(images: tf.Tensor, scale: float) -> tf.Tensor:
    """

    scale: float in range [0, image_resolution]
    """

    # add the blurring:

    h, w, c = get_image_dims(images)
    full_resolution = tf.cast(tf.math.maximum(h, w), tf.float32)

    # Ensure maximum element of x is smaller or equal to 1
    # std = math.sqrt(scale)
    std = scale

    kernel_size = appropriate_kernel_size(std)
    # we won't use a kernel bigger than the resolution of the image!
    kernel_size = tf.clip_by_value(kernel_size, 3, full_resolution)
    
    # In case the kernel size was clipped, we make sure to get the right std for that kernel size.
    # If we don't do this, we might end up with a huge kernel, but with high values even at the edges.
    std = appropriate_std(kernel_size)
    std = tf.math.maximum(std, 0.01)
    with tf.device("cpu:0"), tf.variable_scope("gaussian_blur", reuse=tf.AUTO_REUSE):
        tf.summary.scalar("kernel_size", kernel_size)
        tf.summary.scalar("std", std)
        # tf.summary.scalar("scale", scale)

    # Warn the user if the scale given is larger than what is reasonable.
    # with tf.control_dependencies([tf.print("scale:", scale, "std:", std, "kernel_size:", kernel_size)]):
    return gaussian_blur(images, std, kernel_size)



# @tf.function
def gaussian_blur(
    image: tf.Tensor,
    std: float,
    kernel_size: int,
):
    """
    Performs gaussian blurring. If not given, the right kernel size is infered for the given std.

    NOTE: Since the gaussian filter is separable, we use a 1d kernel and
    convolve twice (more efficient).
    """
    data_format = get_data_format(image)
    assert data_format in {"NHWC", "NCHW"}, "invalid data format"
    
    h, w, c = get_image_dims(image)

    size = kernel_size

    distribution = tfp.distributions.Normal(0, std)
    vals = distribution.prob(tf.range(-(size//2), (size//2)+1, dtype=float))
    kernel = vals / tf.reduce_sum(vals)
    kernel = tf.identity(kernel, name="gaussian_kernel")
    # summary = tf.summary.image(
    #     "gaussian_kernel",
    #     tf.einsum("i,j->ij", kernel, kernel)[tf.newaxis, :, :, tf.newaxis]
    # )

    # expand the kernel to match the requirements of depthsiwe_conv2d
    kernel = kernel[:, tf.newaxis, tf.newaxis, tf.newaxis]
    kernel_h = tf.tile(kernel, [1, 1, c, 1])
    kernel_v = tf.transpose(kernel_h, [1, 0, 2, 3])
        
    result_1 = tf.nn.depthwise_conv2d(
        image,
        kernel_h,
        strides=[1, 1, 1, 1],
        padding="SAME",
        data_format=data_format,
    )
    # flip the kernel, so it is now vertical
    result_2 = tf.nn.depthwise_conv2d(
        result_1,
        kernel_v,
        strides=[1, 1, 1, 1],
        padding="SAME",
        data_format=data_format,
    )

    return result_2


class GaussianBlur2D(keras.layers.Layer):
    def __init__(
        self,
        std: float,
        data_format="channels_first",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.std = std
        self.data_format = data_format
        
        self.trainable = False
        
#     @tf.function
    def call(self, image: tf.Tensor):
        flipped = tf.transpose(image, [0,2,3,1])
        tf.summary.image("image", flipped)
        return gaussian_blur(
            image,
            std=self.std,
            #data_format=self.data_format,
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    kernel_2d = tf.einsum('i,j->ij', kernel_1, kernel_2)
    # Make data.
    X = np.arange(0, size, 1)
    Y = np.arange(0, size, 1)
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, kernel, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()