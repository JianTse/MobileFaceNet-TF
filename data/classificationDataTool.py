# -*-coding: utf-8 -*-
import os
import tensorflow as tf
from scipy import misc
import numpy as np
import random
import sys
import io
import matplotlib.pyplot as plt

def to_rgb(img):
    if img.ndim < 3:
        h, w = img.shape
        ret = np.empty((h, w, 3), dtype=np.uint8)
        ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
        return ret
    else:
        return img


def augmentation(image, aug_img_size):
    ori_image_shape = tf.shape(image)
    image = tf.image.random_flip_left_right(image)
    # image = tf.image.resize_images(image, [aug_img_size, aug_img_size])
    # image = tf.random_crop(image, ori_image_shape)
    return image


class ClassificationImageData:

    def __init__(self, img_size=112, augment_flag=True, augment_margin=16):
        self.img_size = img_size
        self.augment_flag = augment_flag
        self.augment_margin = augment_margin


    def get_path_label(self, root):
        ids = list(os.listdir(root))
        ids.sort()
        self.cat_num = len(ids)
        id_dict = dict(zip(ids, list(range(self.cat_num))))
        paths = []
        labels = []
        for i in ids:
            cur_dir = os.path.join(root, i)
            fns = os.listdir(cur_dir)
            paths += [os.path.join(cur_dir, fn) for fn in fns]
            labels += [id_dict[i]]*len(fns)
        return paths, labels


    def image_processing(self, img):
        img.set_shape([None, None, 3])
        img = tf.image.resize_images(img, [self.img_size, self.img_size])

        if self.augment_flag :
            augment_size = self.img_size + self.augment_margin
            img = augmentation(img, augment_size)
        
        img = tf.cast(img, tf.float32) / 127.5 - 1

        return img


    def add_record(self, img, label, writer):
        img = to_rgb(img)
        img = misc.imresize(img, [self.img_size, self.img_size]).astype(np.uint8)
        shape = img.shape
        tf_features = tf.train.Features(feature={
            "img": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()])),
            "shape": tf.train.Feature(int64_list=tf.train.Int64List(value=list(shape))),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        })
        tf_example = tf.train.Example(features = tf_features)
        tf_serialized = tf_example.SerializeToString()
        writer.write(tf_serialized)
    
    
    def write_tfrecord_from_folders(self, read_dir, write_path):
        print('write tfrecord from folders...')
        writer = tf.python_io.TFRecordWriter(write_path, options=None)
        paths, labels = self.get_path_label(read_dir)
        assert(len(paths) == len(labels))
        total = len(paths)
        cnt = 0
        for p, l in zip(paths, labels):
            img = misc.imread(p).astype(np.uint8)
            self.add_record(img, l, writer)
            cnt += 1
            print('%d/%d\r' % (cnt, total))
        writer.close()
        print('done![%d/%d]' % (cnt, total))
        print('class num: %d' % self.cat_num)


    def write_tfrecord_from_mxrec(self, read_dir, write_path):
        import mxnet as mx
        print('write tfrecord from mxrec...')
        idx_path = os.path.join(read_dir, 'train.idx')
        bin_path = os.path.join(read_dir, 'train.rec')
        imgrec = mx.recordio.MXIndexedRecordIO(idx_path, bin_path, 'r')
        s = imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        imgidx = list(range(1, int(header.label[0])))
        writer = tf.python_io.TFRecordWriter(write_path, options=None)
        total = len(imgidx)
        cnt = 0
        labels = []
        for i in imgidx:
            img_info = imgrec.read_idx(i)
            header, img = mx.recordio.unpack(img_info)
            l = int(header.label)
            labels.append(l)
            img = io.BytesIO(img)
            img = misc.imread(img).astype(np.uint8)
            self.add_record(img, l, writer)
            cnt += 1
            print('%d/%d\r' % (cnt, total))
        writer.close()
        self.cat_num = len(set(labels))
        print('done![%d/%d]' % (cnt, total))
        print('class num: %d' % self.cat_num)


    def parse_function(self, example_proto):
        dics = {
            'img': tf.FixedLenFeature(shape=(), dtype=tf.string),
            'shape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64),
            'label': tf.FixedLenFeature(shape=(), dtype=tf.int64)
        }
        parsed_example = tf.parse_single_example(example_proto, dics)
        parsed_example['img'] = tf.decode_raw(parsed_example['img'], tf.uint8)
        parsed_example['img'] = tf.reshape(parsed_example['img'], parsed_example['shape'])
        return self.image_processing(parsed_example['img']), parsed_example['label']

    def read_TFRecord(self, filenames):
        dataset = tf.data.TFRecordDataset(filenames, buffer_size=256<<20)
        return dataset.map(self.parse_function, num_parallel_calls=8)

    def parse_function_xj(self, example_proto):
        dics = {
            'image_raw': tf.FixedLenFeature(shape=(), dtype=tf.string),
            'label': tf.FixedLenFeature(shape=(), dtype=tf.int64)
        }
        parsed_example = tf.parse_single_example(example_proto, dics)
        parsed_example['image_raw'] = tf.decode_raw(parsed_example['image_raw'], tf.uint8)
        parsed_example['image_raw'] = tf.reshape(parsed_example['image_raw'], [self.img_size, self.img_size, 3])
        return self.image_processing(parsed_example['image_raw']), parsed_example['label']

    def read_TFRecord_xj(self, filenames):
        dataset = tf.data.TFRecordDataset(filenames, buffer_size=256<<20)
        return dataset.map(self.parse_function_xj, num_parallel_calls=8)


def  test_one_shot_inter():
    # train_data = 'E:/work/data/faceRecognize/insightface_faces_umd_ms1m_vgg_112/faces_ms1m_112x112/tran.tfrecords'
    train_data = 'E:/work/data/faceRecognize/insightface_faces_umd_ms1m_vgg_112/faces_ms1m_112x112_train/3/train.tfrecords'

    cid = ClassificationImageData(img_size=112, augment_flag=True, augment_margin=16)
    train_dataset = cid.read_TFRecord_xj(train_data).shuffle(100).repeat().batch(4)
    train_iterator = train_dataset.make_one_shot_iterator()
    next_element = train_iterator.get_next()
    #train_images, train_labels = train_iterator.get_next()
    #train_images = tf.identity(train_images, 'image_raw')
    #train_labels = tf.identity(train_labels, 'label')
    with tf.Session() as sess:
        for i in range(11):
            #train_images, train_labels = sess.run(train_images, train_labels)
            train_images, train_labels = sess.run(next_element)
            print('shape:{},tpye:{}'.format(train_images[0].shape, train_images[0].dtype))
            # show_image("image:%d" % (label), image)
            plt.imshow(train_images[0])
            plt.axis('on')  # 关掉坐标轴为 off
            plt.title("image:%d" % (train_labels[0]))  # 图像题目
            plt.show()

def  test_initializable_iterator():
    # train_data = 'E:/work/data/faceRecognize/insightface_faces_umd_ms1m_vgg_112/faces_ms1m_112x112/tran.tfrecords'
    train_data = 'E:/work/data/faceRecognize/insightface_faces_umd_ms1m_vgg_112/faces_ms1m_112x112_train/3/train.tfrecords'

    cid = ClassificationImageData(img_size=112, augment_flag=True, augment_margin=16)
    train_dataset = cid.read_TFRecord_xj(train_data).shuffle(100).repeat().batch(4)
    train_iterator = train_dataset.make_initializable_iterator()
    next_element = train_iterator.get_next()
    with tf.Session() as sess:
        sess.run(train_iterator.initializer)
        for i in range(11):
            train_images, train_labels = sess.run(next_element)
            print('shape:{},tpye:{}'.format(train_images[0].shape, train_images[0].dtype))
            # show_image("image:%d" % (label), image)
            plt.imshow(train_images[0])
            plt.axis('on')  # 关掉坐标轴为 off
            plt.title("image:%d" % (train_labels[0]))  # 图像题目
            plt.show()

if __name__ == '__main__':
    #test_one_shot_inter()
    test_initializable_iterator()