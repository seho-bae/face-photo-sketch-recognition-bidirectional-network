import os
import numpy as np
import tensorflow as tf
import random


def read_dirs_with_txt(input_dirs, input_txt, num_style=0, concat_styles=True, num_label=None):
    with tf.device('/cpu:0'):
        print(input_txt)
        f = open(input_txt, 'r')
        # Create a list of filenames
        filenames = []
        labels = []
        while True:
            line = f.readline().split()
            if not line: break
            filenames.append(line[0])
            if num_label != None:
                assert int(line[1]) < num_label
            labels.append(int(line[1]))

        # Check imgs
        img_filenames = []
        img_filedirs = []
        img_identities = []
        if not concat_styles:
            for i in range(len(input_dirs)):
                img_filedirs.append([])
                img_identities.append([])

        for i in range(len(filenames)):
            filename = filenames[i]
            label = labels[i]
            if (filename[-4:] == '.jpg') or (filename[-4:] == '.JPG') or (filename[-5:] == '.jpeg') or (filename[-4:] == '.png') or (filename[-4:] == '.PNG'):
                if not os.path.exists(input_dirs[0] + '/' + filename):
                    filename = filename[:-4] + '.png'
                if (num_style > 1) and (not concat_styles):
                    img_filenames.append(filename)
                    for j in range(len(input_dirs)):
                        img_filedirs[j].append(input_dirs[j] + '/' + filename)
                        img_identities[j].append(label)
                else:
                    for j in range(len(input_dirs)):
                        img_filenames.append(filename)
                        img_filedirs.append(input_dirs[j] + '/' + filename)
                        img_identities.append(label)
            else:
                print(filename)
                exit()

    return img_filedirs, img_filenames, img_identities


def load_imgs(filedir_queue, img_size, channel=3, name='', paired=False):
    with tf.device('/cpu:0'):
        # Create a reader for the filequeue
        #reader = tf.WholeFileReader()
        # Read in the files
        #_, image_file = reader.read(filedir_queue)
        # Convert the Tensor(of type string) to representing the Tensor of type uint8
        # and shape [height, width, channels] representing the images
        image = tf.image.decode_image(filedir_queue, channels=channel)
        # set shape of image (for next process)
        if paired:
            image.set_shape([img_size, 2*img_size, channel])
        else:
            image.set_shape([img_size, img_size, channel])

    return image


def preprocess(image, img_size=128, channel=3, flip=False, crop_size=None, random_crop=False, padding_size=None, paired=False, seed=None, is_label=False):
    with tf.device('/cpu:0'):
        if paired:
            # normalize
            image = tf.subtract(tf.to_float(image), 127.5)
            image = tf.divide(image, 127.5)
            # divide image A and B
            imgA, imgB = tf.split(image, 2, axis=1)
            # padding
            if (padding_size != None) and (padding_size != False) and (padding_size != img_size):
                imgA = tf.image.resize_image_with_crop_or_pad(imgA, padding_size, padding_size)
                imgB = tf.image.resize_image_with_crop_or_pad(imgB, padding_size, padding_size)
            # crop
            if seed == None:
                seed = random.randint(0, 2**31 - 1)
            if (crop_size != None) and (crop_size != False) and (crop_size != padding_size):
                if random_crop:
                    imgA = tf.random_crop(imgA, [crop_size, crop_size, channel], seed=seed)
                    imgB = tf.random_crop(imgB, [crop_size, crop_size, channel], seed=seed)
                else:
                    imgA = tf.image.resize_image_with_crop_or_pad(imgA, crop_size, crop_size)
                    imgB = tf.image.resize_image_with_crop_or_pad(imgB, crop_size, crop_size)
                imgA.set_shape([crop_size, crop_size, channel])
                imgB.set_shape([crop_size, crop_size, channel])
            # flip
            if flip == True:
                imgA = tf.image.random_flip_left_right(imgA, seed=seed)
                imgB = tf.image.random_flip_left_right(imgB, seed=seed)
            return imgA, imgB
        else:
            # normalize
            if is_label:
                image = tf.divide(image, 255)
            else:
                image = tf.subtract(tf.to_float(image), 127.5)
                image = tf.divide(image, 127.5)
            # padding
            if (padding_size != None) and (padding_size != False) and (padding_size != img_size):
                image = tf.image.resize_image_with_crop_or_pad(image, padding_size, padding_size)
            # crop
            if seed == None:
                seed = random.randint(0, 2**31 - 1)
            if (crop_size != None) and (crop_size != False) and (crop_size != padding_size):
                if random_crop:
                    image = tf.random_crop(image, [crop_size, crop_size, channel], seed=seed)
                else:
                    image = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
                image.set_shape([crop_size, crop_size, channel])
            # flip
            if flip == True:
                image = tf.image.random_flip_left_right(image, seed=seed)
            # other argumentations
            #image = tf.image.random_brightness(image, max_delta=0.2)
            #image = tf.image.random_contrast(image, lower=0.2, upper=1.2)
            #tf.image.random_hue(image, max_delta=0.5)
            return image   


def load_photo_sketch(input_dir, photo_txt, sketch_txt, style_list, queue_capacity, img_size, num_style,
                      photo_dim=1, sketch_dim=1, flip=True, crop_size=256, random_crop=True, padding_size=None,
                      concat_style_batches=False, num_identity=None, log_dir=None):
    with tf.device('/cpu:0'):
        seed = random.randint(0, 2**31 - 1)    #for random crop
        photo_dir = [input_dir+'/photo']
        print(photo_dir)
        sketch_dirs = []
        assert num_style == len(style_list), 'number of style error'
        for i in range(num_style):
            sketch_dirs.append(input_dir+'/'+style_list[i])
        print(sketch_dirs)

        # read photo directories
        photo_filedirs, photo_filenames, photo_identities = read_dirs_with_txt(photo_dir, photo_txt,  0, num_label=num_identity)
        photo_num = len(photo_filedirs)
        print('number of photo: %d' % photo_num)
        # read sketch directories
        sketch_filedirs, sketch_filenames, sketch_identities = read_dirs_with_txt(sketch_dirs, sketch_txt, num_style, concat_style_batches, num_label=num_identity)
        if concat_style_batches:
            sketch_num = len(sketch_filedirs)
        else:
            sketch_num = num_style*len(sketch_filedirs[0])
        print('number of sketch: %d' % sketch_num)

        # print inputs
        if log_dir != None:
            txtfile = open(log_dir, 'w')
            print('#photos=============================================', file=txtfile)
            print('number of photos: %d' % photo_num, file=txtfile)
            for k in range(len(photo_filedirs)):
                print(photo_filedirs[k], file=txtfile)
            print('#sketches===========================================', file=txtfile)
            print('number of sketches: %d' % sketch_num, file=txtfile)
            for k in range(len(sketch_filedirs)):
                print(sketch_filedirs[k], file=txtfile)
            txtfile.close()

        # convert to tensor
        photo_filedirs = tf.convert_to_tensor(photo_filedirs, dtype=tf.dtypes.string)
        photo_identities = tf.convert_to_tensor(photo_identities, dtype=tf.dtypes.int32)
        photo_filenames = tf.convert_to_tensor(photo_filenames, dtype=tf.dtypes.string)
        sketch_filenames = tf.convert_to_tensor(sketch_filenames, dtype=tf.dtypes.string)
        if concat_style_batches:
            sketch_filedirs = tf.convert_to_tensor(sketch_filedirs, dtype=tf.dtypes.string)
            sketch_identities = tf.convert_to_tensor(sketch_identities, dtype=tf.dtypes.int32)
        else:
            tmp_filedirs = sketch_filedirs
            tmp_identities = sketch_identities
            sketch_filedirs = []
            sketch_identities = []
            for i in range(num_style):
                sketch_filedirs.append(tf.convert_to_tensor(tmp_filedirs[i], dtype=tf.dtypes.string))
                sketch_identities.append(tf.convert_to_tensor(tmp_identities[i], dtype=tf.dtypes.int32))

        # input producer
        if concat_style_batches:
            photo_filedir, photo_identity, photo_name, sketch_filedir, sketch_identity, sketch_name = \
                tf.train.slice_input_producer([photo_filedirs, photo_identities, photo_filenames, sketch_filedirs, sketch_identities, sketch_filenames],
                                              shuffle=False, capacity=queue_capacity, name='photo_filename_queue')
        else:
            tmp_queues = tf.train.slice_input_producer([photo_filedirs, photo_identities, photo_filenames, sketch_filenames] + sketch_filedirs + sketch_identities,
                                                       shuffle=False, capacity=queue_capacity, name='photo_filename_queue')
            photo_filedir = tmp_queues[0]
            photo_identity = tmp_queues[1]
            photo_name = tmp_queues[2]
            sketch_name = tmp_queues[3]
            sketch_filedir = tmp_queues[4:4 + num_style]
            sketch_identity = tmp_queues[4 + num_style:]

        # photo
        photo_file = tf.read_file(photo_filedir)
        photo_decoded = tf.image.decode_image(photo_file, channels=photo_dim)
        photo_decoded.set_shape([img_size, img_size, photo_dim])
        photo = preprocess(photo_decoded, img_size, photo_dim, flip, crop_size, random_crop, padding_size, seed=seed)
        # sketches
        if concat_style_batches:
            sketch_file = tf.read_file(sketch_filedir)
            sketch_decoded = tf.image.decode_image(sketch_file, channels=sketch_dim)
            sketch_decoded.set_shape([img_size, img_size, sketch_dim])
            sketch = preprocess(sketch_decoded, img_size, sketch_dim, flip, crop_size, random_crop, padding_size, seed=seed)
        else:
            sketch = []
            for i in range(num_style):
                sketch_file = tf.read_file(sketch_filedir[i])
                sketch_decoded = tf.image.decode_image(sketch_file, channels=sketch_dim)
                sketch_decoded.set_shape([img_size, img_size, sketch_dim])
                sketch_i = preprocess(sketch_decoded, img_size, sketch_dim, flip, crop_size, random_crop, padding_size, seed=seed)
                sketch.append(sketch_i)

        return photo, photo_identity, photo_name, photo_num, sketch, sketch_identity, sketch_name, sketch_num


def load_photo(input_dir, photo_txt, queue_capacity, img_size, photo_dim=1, flip=True, crop_size=256, random_crop=True,
               padding_size=None, num_identity=None, log_dir=None):
    with tf.device('/cpu:0'):
        seed = random.randint(0, 2**31 - 1)    #for random crop
        photo_dir = [input_dir]
        print(photo_dir)

        # read photo directories
        photo_filedirs, photo_filenames, photo_identities = read_dirs_with_txt(photo_dir, photo_txt,  0, num_label=num_identity)
        photo_num = len(photo_filedirs)
        print('number of photo: %d' % photo_num)

        # print inputs
        if log_dir != None:
            txtfile = open(log_dir, 'w')
            print('#photos=============================================', file=txtfile)
            print('number of photos: %d' % photo_num, file=txtfile)
            for k in range(len(photo_filedirs)):
                print(photo_filedirs[k], file=txtfile)

        # convert to tensor
        photo_filedirs = tf.convert_to_tensor(photo_filedirs, dtype=tf.dtypes.string)
        photo_identities = tf.convert_to_tensor(photo_identities, dtype=tf.dtypes.int32)
        photo_filenames = tf.convert_to_tensor(photo_filenames, dtype=tf.dtypes.string)

        # input producer
        photo_filedir, photo_identity, photo_name = \
            tf.train.slice_input_producer([photo_filedirs, photo_identities, photo_filenames], shuffle=False,
                                          capacity=queue_capacity, name='photo_filename_queue')
        # photo
        photo_file = tf.read_file(photo_filedir)
        photo_decoded = tf.image.decode_image(photo_file, channels=photo_dim)
        photo_decoded.set_shape([img_size, img_size, photo_dim])
        photo = preprocess(photo_decoded, img_size, photo_dim, flip, crop_size, random_crop, padding_size, seed=seed)

        return photo, photo_identity, photo_name, photo_num


def photo_sketch_batch_inputs(input_dir, photo_txt, sketch_txt, num_identity, num_style, style_list, batch_size,
                              img_size=256, name='', photo_dim=3, sketch_dim=3, flip=False, crop_size=None,
                              padding_size=None, random_crop=False, train_mode='train_gan', concat_sketch_styles=False, log_dir=None):
    with tf.device('/cpu:0'):
        if train_mode == 'test_gan':
            queue_capacity = 4 * batch_size
        elif train_mode == 'train_gan':
            queue_capacity = 16 * batch_size
        else:
            assert False, 'input_data: train_mode Error'

        # load images and preprocessing
        # photo dir = input_dir+'/photo'
        if (train_mode == 'test') or (train_mode == 'test_gan'):
            photos, photo_identity, photo_names, photo_num, sketches, sketch_identity, sketch_names, sketch_num = load_photo_sketch(
                input_dir, photo_txt, sketch_txt, style_list, queue_capacity, img_size, num_style,
                photo_dim, sketch_dim, flip=False, crop_size=crop_size, random_crop=False,
                padding_size=padding_size, concat_style_batches=concat_sketch_styles, log_dir=log_dir)
        else:
            photos, photo_identity, photo_names, photo_num, sketches, sketch_identity, sketch_names, sketch_num = load_photo_sketch(
                input_dir, photo_txt, sketch_txt, style_list, queue_capacity, img_size, num_style,
                photo_dim, sketch_dim, flip=flip, crop_size=crop_size, random_crop=random_crop,
                padding_size=padding_size, concat_style_batches=concat_sketch_styles, num_identity=num_identity,
                log_dir=log_dir)
            # assert sketch_num == photo_num*num_style, 'Num of data problem'

        # make batches
        if (train_mode=='train_gan'):
            seed = random.randint(0, 2**31 - 1)  # use same seed for every batches
            if concat_sketch_styles:
                photo_batch, photo_identity_batch, photo_name_batch, sketch_batch, sketch_identity_batch, sketch_name_batch = \
                    tf.train.shuffle_batch([photos, photo_identity, photo_names, sketches, sketch_identity, sketch_names],
                                           batch_size=batch_size, num_threads=1, capacity=queue_capacity,
                                           min_after_dequeue=int(queue_capacity/2), seed=seed, name=name+'batch_queue')
            else:
                tmp_batches = tf.train.shuffle_batch([photos, photo_identity, photo_names, sketch_names] + sketches + sketch_identity,
                                                     batch_size=batch_size, num_threads=1, capacity=queue_capacity,
                                                     min_after_dequeue=int(queue_capacity/2), seed=seed, name=name+'batch_queue')
                photo_batch = tmp_batches[0]
                photo_identity_batch = tmp_batches[1]
                photo_name_batch = tmp_batches[2]
                sketch_name_batch = tmp_batches[3]
                sketch_batch = tmp_batches[4:4+num_style]
                sketch_identity_batch = tmp_batches[4+num_style:]
        elif (train_mode=='test_gan'):
            if concat_sketch_styles:
                photo_batch, photo_identity_batch, photo_name_batch, sketch_batch, sketch_identity_batch, sketch_name_batch = \
                    tf.train.batch([photos, photo_identity, photo_names, sketches, sketch_identity, sketch_names],
                                   batch_size=batch_size, num_threads=1, capacity=queue_capacity,
                                   name=name+'batch_queue')
            else:
                tmp_batches = tf.train.batch([photos, photo_identity, photo_names, sketch_names] + sketches + sketch_identity,
                                             batch_size=batch_size, num_threads=1, capacity=queue_capacity,
                                             name=name+'batch_queue')
                photo_batch = tmp_batches[0]
                photo_identity_batch = tmp_batches[1]
                photo_name_batch = tmp_batches[2]
                sketch_name_batch = tmp_batches[3]
                sketch_batch = tmp_batches[4:4+num_style]
                sketch_identity_batch = tmp_batches[4+num_style:]
        else:
            assert False, 'input_data: train_mode Error'

        '''
        dict={'ph':photo_batch,'ph_id':photo_identity_batch,'phn':photo_name_batch}
        with tf.Session() as sess:
            # Coordinate the loading of image files.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            print("Queue started")

            result=sess.run(dict)
            print(result)

            coord.request_stop()
            coord.join(threads)
            sess.close()

        exit()
        '''

        return photo_batch, sketch_batch, photo_identity_batch, sketch_identity_batch, photo_name_batch, sketch_name_batch, photo_num, sketch_num


def photo_batch_inputs(input_dir, photo_txt, num_identity, batch_size, img_size=256, name='', photo_dim=3, flip=False,
                       crop_size=None, padding_size=None, random_crop=False, train_mode='test_gallery', log_dir=None):
    with tf.device('/cpu:0'):
        if train_mode == 'test_gallery':
            queue_capacity = 4 * batch_size
        else:
            assert False, 'input_data: photo train_mode Error'

        # load images and preprocessing
        # photo dir = input_dir+'/photo'
        if (train_mode == 'test') or (train_mode == 'test_gallery'):
            photos, photo_identity, photo_names, photo_num = load_photo(input_dir, photo_txt, queue_capacity, img_size,
                                                                        photo_dim, flip=False, crop_size=crop_size,
                                                                        random_crop=False, padding_size=padding_size,
                                                                        num_identity=num_identity, log_dir=log_dir)
        else:
            photos, photo_identity, photo_names, photo_num = load_photo(input_dir, photo_txt, queue_capacity, img_size,
                                                                        photo_dim, flip=flip, crop_size=crop_size,
                                                                        random_crop=random_crop, padding_size=padding_size,
                                                                        num_identity=num_identity, log_dir=log_dir)

        # make batches
        if train_mode == 'test_gallery':
            photo_batch, photo_identity_batch, photo_name_batch = tf.train.batch([photos, photo_identity, photo_names],
                                                                                 batch_size=batch_size, num_threads=1,
                                                                                 capacity=queue_capacity, name=name+'batch_queue')
        else:
            assert False, 'Photo input_data: train_mode Error'

        '''
        dict={'ph':photo_batch,'ph_id':photo_identity_batch,'phn':photo_name_batch}
        with tf.Session() as sess:
            # Coordinate the loading of image files.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            print("Queue started")

            result=sess.run(dict)
            print(result)

            coord.request_stop()
            coord.join(threads)
            sess.close()

        exit()
        '''

        return photo_batch, photo_identity_batch, photo_name_batch, photo_num