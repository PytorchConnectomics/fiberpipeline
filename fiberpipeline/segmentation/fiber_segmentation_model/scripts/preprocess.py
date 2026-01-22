import os
import cv2
import argparse
import numpy as np
from glob import glob
import tifffile as tf
from tqdm import tqdm
from connectomics.data.utils.data_io import readvol, savevol

def remove_outliers(vol, clip_percentile):
    high, low = np.percentile(vol, [100-clip_percentile, clip_percentile])
    vol = np.clip(vol, low, high)
    return vol

def normalize(vol, do_uint8=True):
    vol = vol.astype(float)
    vol -= vol.min()
    vol /= vol.max()
    if do_uint8:
        vol = (vol * 255).astype(np.uint8)
    return vol

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-d', default='/projects/weilab/dataset/barcode/train_r2/PT37', type=str)
    parser.add_argument('--input_pattern', '-i', default='*.tif', type=str)
    parser.add_argument('--clip_percentile', default=5, type=int)
    parser.add_argument('--do_uint8', default=False, type=bool)
    args = parser.parse_args()

    for fname in (pbar := tqdm(sorted(glob(os.path.join(args.data_dir, args.input_pattern))))):
        suffix = fname.split('.')[-1]
        if suffix in ['tif', 'tiff']:
            data = tf.imread(fname)
        elif suffix in ['h5']:
            data = readvol(fname)
        else:
            raise RuntimeError('suffix doesn\'t match tif or h5')
        
        # the matrix stored in the variable 'data' should be 3D
        # if 4D, assume that 'data' follows the shape (num_sample, num_channel, y, x)
        # cell segmentation data should be in num_channel = 1
        if len(data.shape) == 4:
            backup = data
            data = data[:, 1]

        data = remove_outliers(data, args.clip_percentile)
        data = (normalize(data,  do_uint8=args.do_uint8) * 255).astype(np.uint16)

        # if the data was originally 4D, save it as such
        if 'backup' in locals():
            backup[:, 1] = data
            data = backup
            del backup

        fname = fname.replace('nov_11', 'nov_12')
        if suffix in ['tif', 'tiff']:
            tf.imwrite(fname.replace(f'.{suffix}', f'-preprocessed.{suffix}'), data)
        elif suffix in ['h5']:
            savevol(fname.replace(f'.{suffix}', f'-preprocessed.{suffix}'), data)
        else:
            raise RuntimeError('suffix doesn\'t match tif or h5')
