import imageio
from scipy.misc import imsave
import numpy as np
import os
import moviepy.editor as mp

# current path
cur_path = os.getcwd()


def get_frames(filename):
    """Get frames from the video
    Input : path to the video file, Ex : Video/main.mp4
    Output : frame names,           Ex : nar00001, nar00002, ....

    """
    duration = mp.VideoFileClip(filename).duration  # gives the duration of the video
    vid = imageio.get_reader(filename, 'ffmpeg')
    n_frames = int(duration) * 30  # Assuming a 30 fps video
    nums = np.arange(1, n_frames, 200)  # skip 4 frames
    num = np.arange(len(nums))
    vid.get_meta_data()
    for x, y in zip(num, nums):
        try:
            image = vid.get_data(y)
            fname = 'nar%05d.png' % x
            imsave(fname, image)
        except:
            pass

    # Compute the list of all images
    frames = os.listdir(cur_path)
    # remove all the useless files and directories from the list
    frames = [i for i in frames if 'nar' in i]
    # sort all the frames in the list
    frames.sort()
    return frames
