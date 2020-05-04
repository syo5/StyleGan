# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
import pickle
import numpy as np
from PIL import Image
import dnnlib
import dnnlib.tflib as tflib
import config

def main():
    # Initialize TensorFlow.
    tflib.init_tf()

    # Load pre-trained network.
    fpath = 'C:\\Users\\SS\\Desktop\\stylegans-pytorch-master\\karras2019stylegan-ffhq-1024x1024.pkl' # karras2019stylegan-ffhq-1024x1024.pkl
    with open(fpath, mode='rb') as f:
        _G, _D, Gs = pickle.load(f)
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

    # Print network details.
    Gs.print_layers()

    # Pick latent vector.
    rnd = np.random.RandomState(6) #5
    latents_=[]
    a=17
    b=18
    for i in range(1,101,4):
        latents = rnd.randn(1, Gs.input_shape[1])
        latents_.append(latents)

    for j in range(25):
        latents_mean=j/25*latents_[a]+(1-j/25)*latents_[b]
        # Generate image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents_mean, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
        # Save image.
        os.makedirs(config.result_dir, exist_ok=True)
        png_filename = os.path.join(config.result_dir, 'example{}.png'.format(j))
        Image.fromarray(images[0], 'RGB').save(png_filename)

    s=50
    images = []
    for i in range(25):
        im = Image.open(config.result_dir+'/example'+str(i)+'.png')
        im =im.resize(size=(512, 512), resample=Image.NEAREST)
        images.append(im)
    for i in range(24,0,-1):
        im = Image.open(config.result_dir+'/example'+str(i)+'.png')
        im =im.resize(size=(512, 512), resample=Image.NEAREST)
        images.append(im)

    images[0].save(config.result_dir+'/example{}_{}.gif'.format(a,b), save_all=True, append_images=images[1:s], duration=100*2, loop=0)
if __name__ == "__main__":
    main()
