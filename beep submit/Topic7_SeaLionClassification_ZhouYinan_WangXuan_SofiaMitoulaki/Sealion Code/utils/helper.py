import cv2
import glob
import skimage.feature
import pandas as pd
import numpy as np
import os


bad_images = [3, 7, 9, 21, 30, 34, 71, 81, 89, 97, 151, 184, 215, 234, 242, 268, 290, 311, 331, 344, 380, 384, 406, 421,
              469, 475, 490, 499, 507,
              530, 531, 605, 607, 614, 621, 638, 644, 687, 712, 721, 767, 779, 781, 794, 800, 811, 839, 840, 869, 882,
              901, 903, 905, 909, 913, 927, 946]

sealion_classes = ['adult_males', 'subadult_males', 'adult_females', 'juveniles', 'pups']


def get_sealion_coords(path):
    """
    Generate coods.csv file which contains each sea lion coordinates
    :param path: data path
    :return: None
    """
    train_path = path + 'Train/'
    train_dotted_path = path + 'TrainDotted/'

    tids = list(map(lambda x: int(os.path.basename(x)[:-4]), glob.glob(train_path + '*.jpg')))
    tids = sorted(list(filter(lambda x: x not in bad_images, tids)))

    print('there are totally {} images'.format(len(tids)))

    res = []

    for i, tid in enumerate(tids):

        # read the Train and Train Dotted images
        image_1 = cv2.imread(train_dotted_path + '{}.jpg'.format(tid))
        image_2 = cv2.imread(train_path + '{}.jpg'.format(tid))

        # absolute difference between Train and TrainDotted
        image_3 = cv2.absdiff(image_1, image_2)

        # mask out blackened regions from Train Dotted
        mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
        mask_1[mask_1 < 20] = 0
        mask_1[mask_1 > 0] = 255

        mask_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
        mask_2[mask_2 < 20] = 0
        mask_2[mask_2 > 0] = 255

        image_3 = cv2.bitwise_or(image_3, image_3, mask=mask_1)
        image_3 = cv2.bitwise_or(image_3, image_3, mask=mask_2)

        # convert to grayscale to be accepted by skimage.feature.blob_log
        image_3 = cv2.cvtColor(image_3, cv2.COLOR_BGR2GRAY)

        # detect blobs
        blobs = skimage.feature.blob_log(image_3, min_sigma=3, max_sigma=4, num_sigma=1, threshold=0.02)

        res_dic = {
            'adult_males': [],
            'subadult_males': [],
            'adult_females': [],
            'juveniles': [],
            'pups': []
        }

        for blob in blobs:
            # get the coordinates for each blob
            y, x, s = blob
            # get the color of the pixel from Train Dotted in the center of the blob
            b, g, r = image_1[int(y)][int(x)][:]

            # decision tree to pick the class of the blob by looking at the color in Train Dotted
            if r > 200 and b < 50 and g < 50:  # RED
                res_dic['adult_males'].append((int(x), int(y)))
            elif r > 200 and b > 200 and g < 50:  # MAGENTA
                res_dic['subadult_males'].append((int(x), int(y)))
            elif r < 100 and b < 100 and 150 < g < 200:  # GREEN
                res_dic['pups'].append((int(x), int(y)))
            elif r < 100 and b > 100 and g < 100:  # BLUE
                res_dic['juveniles'].append((int(x), int(y)))
            elif r < 150 and b < 50 and g < 100:  # BROWN
                res_dic['adult_females'].append((int(x), int(y)))

        for j, _class in enumerate(sealion_classes):
            for (y, x) in res_dic[_class]:
                res.append([tid, j, x, y, _class])

        print('\r{} image completes...'.format(i+1), end='')

    print()
    coords_df = pd.DataFrame(data=res, columns=['tid', 'cls', 'row', 'col', 'class'])
    coords_df.to_csv(path + 'coords.csv', index=False)
    print("coords_csv saved!")


def get_sl_patches(path, tids, size, mode):
    coords_df = pd.read_csv(path + 'coords.csv')
    coords_df = coords_df[coords_df.tid.isin(tids)]
    print('extracting sea lion patches from {} images'.format(len(tids)))

    for i in range(len(coords_df)):
        tid = coords_df.iloc[i].tid
        row = coords_df.iloc[i].row
        col = coords_df.iloc[i].col
        cls = coords_df.iloc[i].cls

        img_path = path + 'Train/{}.jpg'.format(tid)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        patch = img[row - size//2:row + size//2, col - size//2:col + size//2, :]
        if patch.shape != (size, size, 3):
            continue

        if mode == 'train':
            save_path = path + 'Patch/train/{}/'.format(sealion_classes[cls])
            idx = len(glob.glob(save_path + '*.jpg'))
            cv2.imwrite(save_path+'{}.jpg'.format(idx), cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        elif mode == 'valid':
            save_path = path + 'Patch/valid/{}/'.format(sealion_classes[cls])
            idx = len(glob.glob(save_path + '*.jpg'))
            cv2.imwrite(save_path + '{}.jpg'.format(idx), cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        elif mode == 'test':
            save_path = path + 'Patch/test/{}/'.format(sealion_classes[cls])
            idx = len(glob.glob(save_path + '*.jpg'))
            cv2.imwrite(save_path + '{}.jpg'.format(idx), cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))

        print('\r%d patches completes...' % (i + 1), end='')
    print()


def get_bg_patches(path, tids, size, mode):

    coords_df = pd.read_csv(path + 'coords.csv')
    coords_df = coords_df[coords_df.tid.isin(tids)]
    print('extracting background patches from {} images'.format(len(tids)))

    for i, tid in enumerate(tids):
        rows = coords_df[coords_df.tid == tid].row
        cols = coords_df[coords_df.tid == tid].col
        coords = list(zip(rows, cols))

        img_path = path + 'Train/{}.jpg'.format(tid)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for r in range(img.shape[0] // size):
            for c in range(img.shape[1] // size):
                center = (r*size + size//2, c*size + size//2)
                flag = True
                for sl in coords:
                    dist = (center[0] - sl[0]) ** 2 + (center[1] - sl[1]) ** 2
                    if dist < size * size * 2:
                        flag = False
                        break
                if flag:
                    patch = img[center[0] - size//2:center[0] + size//2, center[1] - size//2:center[1] + size//2, :]
                    if patch.shape != (size, size, 3):
                        continue

                    if mode == 'train':
                        save_path = path + 'Patch/train/backgrounds/'
                        idx = len(glob.glob(save_path + '*.jpg'))
                        cv2.imwrite(save_path + '{}.jpg'.format(idx), cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
                    elif mode == 'test':
                        save_path = path + 'Patch/test/backgrounds/'
                        idx = len(glob.glob(save_path + '*.jpg'))
                        cv2.imwrite(save_path + '{}.jpg'.format(idx), cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
                    elif mode == 'valid':
                        save_path = path + 'Patch/valid/backgrounds/'
                        idx = len(glob.glob(save_path + '*.jpg'))
                        cv2.imwrite(save_path + '{}.jpg'.format(idx), cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))



def get_patch_info(path):

    names = ['adult_males', 'subadult_males', 'adult_females', 'juveniles', 'pups', 'backgrounds']
    res_dic = {
        'adult_males': 0,
        'subadult_males': 0,
        'adult_females': 0,
        'juveniles': 0,
        'pups': 0,
        'backgrounds': 0
    }

    # for train
    for name in names:
        res_dic[name] = len(glob.glob(path + 'Patch/train/{}/*.jpg'.format(name)))
    print('train data:')
    print('adult_males {}, subadult_males {}, adult_females {}, juveniles {}, pups {}, backgrounds {}'
          .format(res_dic['adult_males'], res_dic['subadult_males'], res_dic['adult_females'],
                  res_dic['juveniles'], res_dic['pups'], res_dic['backgrounds']))

    # for valid
    for name in names:
        res_dic[name] = len(glob.glob(path + 'Patch/valid/{}/*.jpg'.format(name)))
    print('valid data:')
    print('adult_males {}, subadult_males {}, adult_females {}, juveniles {}, pups {}, backgrounds {}'
          .format(res_dic['adult_males'], res_dic['subadult_males'], res_dic['adult_females'],
                  res_dic['juveniles'], res_dic['pups'], res_dic['backgrounds']))

    # for test
    for name in names:
        res_dic[name] = len(glob.glob(path + 'Patch/test/{}/*.jpg'.format(name)))
    print('test data:')
    print('adult_males {}, subadult_males {}, adult_females {}, juveniles {}, pups {}, backgrounds {}'
          .format(res_dic['adult_males'], res_dic['subadult_males'], res_dic['adult_females'],
                  res_dic['juveniles'], res_dic['pups'], res_dic['backgrounds']))