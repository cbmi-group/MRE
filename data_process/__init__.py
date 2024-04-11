import glob
import os
import numpy as np
import cv2
from data_process.data_augmentation import *



def generate_train_txt():
    absolutepath='/data/ldap_shared/home/s_hjx/PythonProjects/bionetwork'
    txtPath = os.path.join('../datasets', "train_mito.txt")
    file_path = '../mito_dataset_hjx/train/'
    masks_dir = file_path + 'masks_aug/'
    images_dir = file_path + 'images_aug/'
    img_list = glob.glob(os.path.join(images_dir, "*.tif"))
    img_list.sort()
    with open(txtPath, 'w') as f:
        for i, p in enumerate(img_list):
            img_name = os.path.split(p)[-1]
            print("==> Process image: %s." % (img_name))
            f.writelines(absolutepath+images_dir[2:]+ img_name + " " +absolutepath+masks_dir[2:] + img_name + "\n")


def generate_test_txt():
    absolutepath = '/data/ldap_shared/home/s_hjx/PythonProjects/bionetwork'
    txtPath = os.path.join('../datasets', "test_mito.txt")
    file_path = '../mito_dataset_hjx/test/'
    masks_dir = file_path + 'masks/'
    images_dir = file_path + 'images/'

    img_list = glob.glob(os.path.join(images_dir, "*.tif"))
    img_list.sort()
    with open(txtPath, 'w') as f:
        for i, p in enumerate(img_list):
            img_name = os.path.split(p)[-1]
            print("==> Process image: %s." % (img_name))
            f.writelines(absolutepath+images_dir[2:]+ img_name + " " +absolutepath+masks_dir[2:] + img_name + "\n")


if __name__ == '__main__':
    generate_train_txt()
    # file_path = '../nucleus/'
    # masks_dir = file_path + 'masks_train'
    # images_dir = file_path + 'images_train'
    # img_list = glob.glob(os.path.join(images_dir, "*.tif"))
    # img_list.sort()
    #
    # for i, p in enumerate(img_list):
    #     img_name = os.path.split(p)[-1]
    #
    #     print("==> Process image: %s." % (img_name))
    #
    #     now_image = cv2.imread(p, -1)
    #     now_mask = cv2.imread(os.path.join(masks_dir, img_name), -1)
    #     flip_image, flip_mask = flip(now_image, now_mask)
    #
    #     now_image_90, now_mask_90 = rotation_image(now_image, now_mask, angle=90, scale=1)
    #     now_image_180, now_mask_180 = rotation_image(now_image, now_mask, angle=180, scale=1)
    #     now_image_270, now_mask_270 = rotation_image(now_image, now_mask, angle=270, scale=1)
    #
    #     flip_image_90, flip_mask_90 = rotation_image(flip_image, flip_mask, angle=90, scale=1)
    #     flip_image_180, flip_mask_180 = rotation_image(flip_image, flip_mask, angle=180, scale=1)
    #     flip_image_270, flip_mask_270 = rotation_image(flip_image, flip_mask, angle=270, scale=1)
    #
    #     cv2.imwrite(images_dir + '_aug/' + img_name[:-4] + '.tif', now_image)
    #     cv2.imwrite(images_dir + '_aug/' + img_name[:-4] + '_90.tif', now_image_90)
    #     cv2.imwrite(images_dir + '_aug/' + img_name[:-4] + '_180.tif', now_image_180)
    #     cv2.imwrite(images_dir + '_aug/' + img_name[:-4] + '_270.tif', now_image_270)
    #
    #     cv2.imwrite(images_dir + '_aug/flip_' + img_name[:-4] + '.tif', flip_image)
    #     cv2.imwrite(images_dir + '_aug/flip_' + img_name[:-4] + '_90.tif', flip_image_90)
    #     cv2.imwrite(images_dir + '_aug/flip_' + img_name[:-4] + '_180.tif', flip_image_180)
    #     cv2.imwrite(images_dir + '_aug/flip_' + img_name[:-4] + '_270.tif', flip_image_270)
    #
    #     cv2.imwrite(masks_dir + '_aug/' + img_name[:-4] + '.tif', now_mask)
    #     cv2.imwrite(masks_dir + '_aug/' + img_name[:-4] + '_90.tif', now_mask_90)
    #     cv2.imwrite(masks_dir + '_aug/' + img_name[:-4] + '_180.tif', now_mask_180)
    #     cv2.imwrite(masks_dir + '_aug/' + img_name[:-4] + '_270.tif', now_mask_270)
    #
    #     cv2.imwrite(masks_dir + '_aug/flip_' + img_name[:-4] + '.tif', flip_mask)
    #     cv2.imwrite(masks_dir + '_aug/flip_' + img_name[:-4] + '_90.tif', flip_mask_90)
    #     cv2.imwrite(masks_dir + '_aug/flip_' + img_name[:-4] + '_180.tif', flip_mask_180)
    #     cv2.imwrite(masks_dir + '_aug/flip_' + img_name[:-4] + '_270.tif', flip_mask_270)
