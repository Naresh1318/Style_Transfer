from __future__ import print_function
import time
import os
import numpy as np
import keras.backend as K
from video_to_frame import get_frames
from PIL import Image
from argparse import ArgumentParser
from keras.applications import vgg16
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave

# Parameters
height = 512
width = 512


def create_args():
    parser = ArgumentParser()
    parser.add_argument("--content_path", "-c_p",
                        help="path of the content image", required=True)
    parser.add_argument("--style_path", "-s_p",
                        help="path of the style image", required=True)
    parser.add_argument("--content_weight", "-c_w",
                        help="Weighting factor for the content loss", default=0.025, type=float)
    parser.add_argument("--style_weight", "-s_w",
                        help="Weighting factor for the style loss", default=5.0, type=float)
    parser.add_argument("--total_variation_weight", "-t_v_w",
                        help="Weight that reduces the noise in the image", default=1.0, type=float)
    parser.add_argument("--epochs", "-e",
                        help="No of epochs to optimize", default=3, type=int)
    parser.add_argument("--video", "-v",
                        help="Indicate whether you want to perform style transfer on video or an image",
                        action='store_true', dest="video")
    return parser


def preprocess(x):
    """Subtract the mean values of RGB, increase ndim to 4 and convert RGB to BGR"""
    x = x.resize((height, width))
    x = np.asarray(x, dtype='float32')
    x = np.expand_dims(x, axis=0)

    x[:, :, :, 0] -= 103.939
    x[:, :, :, 1] -= 116.779
    x[:, :, :, 2] -= 123.68
    x = x[:, :, :, ::-1]  # RGB to BGR

    return x


def remove_preprocess(x):
    """Add back the subtracted mean values"""
    x = x.reshape((height, width, 3))
    x = x[:, :, ::-1]  # Back to RGB
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def content_loss(content, combined):
    """computes the Euclidean distance between feature representations"""
    return K.sum(K.square(combined - content))


def gram_matrix(x):
    """Compute the gram matrix"""
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss(style, combined):
    """Compute the style loss"""
    style_gram = gram_matrix(style)
    combined_gram = gram_matrix(combined)
    channels = 3
    size = height * width
    return K.sum(K.square(style_gram - combined_gram)) / (4. * (channels ** 2) * (size ** 2))


def total_variation_loss(x):
    """Find the total variation loss which is used to reduce the noise in the generated image"""
    down = K.square(x[:, :height - 1, :width - 1, :] - x[:, 1:, :width - 1, :])
    right = K.square(x[:, :height - 1, :width - 1, :] - x[:, :height - 1, 1:, :])
    return K.sum(K.pow(down + right, 1.25))


def eval_loss_and_grads(x):
    x = x.reshape((1, height, width, 3))
    outs = f_outputs([x])
    loss_value = outs[0]

    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')

    return loss_value, grad_values


# The optimizer used (f_min_l_bfgs_b) requires a separate function to compute loss and gradient
class Evaluator(object):
    def __init__(self):
        self.grad_values = None
        self.loss_value = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.grad_values = grad_values
        self.loss_value = loss_value
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


parser = create_args()
options = parser.parse_args()

# Weights to compute the loss
content_weight = options.content_weight  # 0.025
style_weight = options.style_weight  # 5.0
total_variation_weight = options.total_variation_weight  # 1.0

# Get the style image
style_image_path = options.style_path
style_image = Image.open(style_image_path)
style_array = preprocess(style_image)

# Check whether to style video or an image
if options.video:
    frames = get_frames(options.content_path)
else:
    frames = [options.content_path]

# Training
x = np.random.uniform(0, 255, (1, height, width, 3)) - 128.

# Get the content image
for j, frame in enumerate(frames):
    content_image = Image.open(frame)
    content_array = preprocess(content_image)

    # Tensorflow variables and placeholders for the graph
    content_image = K.variable(content_array)
    style_image = K.variable(style_array)
    combined_image = K.placeholder((1, height, width, 3))

    # Concatenate into a single tensor
    input_tensor = K.concatenate([content_image, style_image, combined_image], axis=0)

    # Load the weights for vgg16
    model = vgg16.VGG16(input_tensor=input_tensor, include_top=False, weights='imagenet')

    # Get all the layers in vgg16
    layers = dict([(layer.name, layer.output) for layer in model.layers])

    loss = K.variable(0.)  # Default dtype = float32

    # Calculate the content loss
    layer_features_content = layers['block2_conv2']
    content_image_features = layer_features_content[0, :, :, :]
    combined_image_features = layer_features_content[2, :, :, :]
    loss += content_weight * content_loss(content_image_features, combined_image_features)

    # Calculate the style loss
    layer_features = ['block1_conv2', 'block2_conv2',
                      'block3_conv3', 'block4_conv3',
                      'block5_conv3']

    for layer in layer_features:
        layer_features_style = layers[layer]
        style_image_features = layer_features_style[1, :, :, :]
        combined_image_features = layer_features_style[2, :, :, :]
        loss += (style_weight / len(layer_features)) * style_loss(style_image_features, combined_image_features)

    # Compute the total variation loss
    loss += total_variation_weight * total_variation_loss(combined_image)

    # Find the gradients
    grads = K.gradients(loss, combined_image)
    outputs = [loss]

    if type(grads) in {list, tuple}:
        outputs += grads
    else:
        outputs.append(grads)

    f_outputs = K.function([combined_image], outputs)

    evaluator = Evaluator()

    epochs = options.epochs

    for i in range(epochs):
        print('Start of iteration', i)
        start_time = time.time()
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                         fprime=evaluator.grads, maxfun=20)
        print('Current loss value:', min_val)
        end_time = time.time()
        print('Iteration %d completed in %ds' % (i, end_time - start_time))
    x = x.reshape((height, width, 3))
    x = x[:, :, ::-1]  # Back to RGB
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = np.clip(x, 0, 255).astype('uint8')
    imsave('combined_dh_{}.jpg'.format(j), Image.fromarray(x))

# execute this linux command to convert the video images to a video
os.system('ffmpeg -f image2 -r 1/5 -i {}%d.jpg -vcodec mpeg4 -y movie.mp4'.format('combined_dh_'))
