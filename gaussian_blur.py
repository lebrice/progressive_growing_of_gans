"""
2D Gaussian Blur Keras layer.
"""
import tensorflow as tf
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


def gaussian_kernel_1d(std, kernel_size):
    x = tf.range(-(kernel_size//2), (kernel_size//2)+1, dtype=float)
    g = tf.exp(- (x**2 / (2 * std**2))) / (tf.sqrt(2 * np.pi) * std)
    # normalize the sum to 1
    g = g / tf.reduce_sum(g)
    return g


def image_at_scale(images: tf.Tensor, scale: float) -> tf.Tensor:
    return blur_images(images, scale)


def limit_scale(images, std, min_kernel_size=3, max_kernel_size=None):
    """
    returns an appropriate set of values for the standard deviation and kernel_size for the given images and the desired standard deviation of the kernel.
    If the standard deviation given would require a larger kernel size than the input image dimensions, the std is reduced to the maximum allowed kernel size.
    """

    kernel_size = appropriate_kernel_size(std)
    # we won't use a kernel bigger than the resolution of the image!
    if max_kernel_size is None:
        h, w, c = get_image_dims(images)
        full_resolution = tf.cast(tf.math.maximum(h, w), tf.float32)

    kernel_size = tf.clip_by_value(kernel_size, min_kernel_size, max_kernel_size)
    
    # In case the kernel size was clipped, we make sure to get the right std for that kernel size.
    # If we don't do this, we might end up with a huge kernel, but with high values even at the edges.
    std = appropriate_std(kernel_size)
    std = tf.math.maximum(std, 0.01)
    return std, kernel_size


def blur_images(images: tf.Tensor, scale: float) -> tf.Tensor:
    """
    Blurr the images using a gaussian blur with the given scale (std).
    """
    std, kernel_size = limit_scale(images, scale)

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
    
    kernel = gaussian_kernel_1d(std, kernel_size)
    kernel = tf.identity(kernel, name="gaussian_kernel")
    # summary = tf.summary.image(
    #     "gaussian_kernel",
    #     tf.einsum("i,j->ij", kernel, kernel)[tf.newaxis, :, :, tf.newaxis]
    # )

    # expand the kernel to match the requirements of depthsiwe_conv2d
    h, w, c = get_image_dims(image)
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
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.std = std
        self.trainable = False
        
#     @tf.function
    def call(self, image: tf.Tensor):
        flipped = tf.transpose(image, [0,2,3,1])
        tf.summary.image("image", flipped)
        return image_at_scale
            image,
            self.std,
        )
