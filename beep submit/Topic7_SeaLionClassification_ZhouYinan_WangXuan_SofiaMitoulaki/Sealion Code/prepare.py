from utils import helper
import pandas as pd
import numpy as np
import os
import glob

path = os.getcwd() + '/data/'

print('* calculating coordinates...')
helper.get_sealion_coords(path)
coords_df = pd.read_csv(path + 'coords.csv')
tids = pd.unique(coords_df.tid)
n_train = int(0.85 * len(tids))
non_test_tids = tids[:n_train]
test_tids = tids[n_train:]
n_valid = int(0.85 * len(non_test_tids))
train_tids = non_test_tids[:n_valid]
valid_tids = non_test_tids[n_valid:]

print('images: train {}, valid {}, test {}'.format(len(train_tids), len(valid_tids), len(test_tids)))

print('* clearing...')
for f in glob.glob(path+'Patch/**/**/*.jpg', recursive=True):
    try:
        os.remove(f)
    except:
        continue

# exit()
print('* extracting patches...')
helper.get_sl_patches(path, train_tids, size=96, mode='train')
helper.get_bg_patches(path, train_tids, size=96, mode='train')
helper.get_sl_patches(path, valid_tids, size=96, mode='valid')
helper.get_bg_patches(path, valid_tids, size=96, mode='valid')
helper.get_sl_patches(path, test_tids, size=96, mode='test')
helper.get_bg_patches(path, test_tids, size=96, mode='test')

helper.get_patch_info(path)

