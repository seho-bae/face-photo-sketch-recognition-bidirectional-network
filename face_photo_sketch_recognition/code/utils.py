import numpy as np
import os
from skimage import io
import html
import tensorflow as tf
import re


def save_examples(img, img_dir, name, num=None):
    # img pixel value: (-1,1)

    if num == None:
        num = len(img)

    img = (img[0:num] + 1) * 127.5
    img = img.astype('uint8')

    if not os.path.exists(img_dir):
        os.mkdir(img_dir)

    if img.shape[3] == 1:
        img = np.resize(img, (num, img.shape[1], img.shape[2]))

    for i in range(num):
        io.imsave(img_dir + '/' + str(name[i]) + '.png', img[i])

    return


def visualize_results(log_dir, result, display_list, folder_list=None):
    if folder_list is None:
        folder_list = display_list
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    for folder in folder_list:
        if not os.path.exists(folder):
            os.mkdir(folder)
    # img names
    name_list = []
    for i in range(len(result['name'])):
        name_list.append(os.path.splitext(result['name'][i].decode('utf-8'))[0])
    # display imgs
    for i in range(len(display_list)):
        display=display_list[i]
        folder=folder_list[i]
        save_examples(result[display], log_dir + '/' + folder, name_list)

    return


def write_html(log_dir, display_list):
    name = os.listdir(log_dir + '/' + display_list[0])
    name.sort(key=natural_keys)

    html_file = open(log_dir + '/results.html', 'w')
    html_file.write('<html><body><table>')
    html_file.write('<tr><th>NAME</th>')
    for folder in display_list:
        html_file.write('<th>'+folder+'</th>')
    html_file.write('</tr>')

    for i in range(len(name)):
        html_file.write('<tr>')
        html_file.write('<td><center>%s</center></td>' % name[i])
        for folder in display_list:
            html_file.write("<td><img src='%s'></td>" % (folder + '/' + name[i]))
        html_file.write('</tr>')

    html_file.write('</table></body></html>')
    html_file.close()

    return


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma)  # window shape [size, size]
    if img1.get_shape()[3] == 3:
        img1 = tf.image.rgb_to_grayscale(img1)
    if img2.get_shape()[3] == 3:
        img2 = tf.image.rgb_to_grayscale(img2)
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = tf.nn.conv2d(img1, window, strides=[1, 1, 1, 1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1, 1, 1, 1], padding='VALID')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                              (sigma1_sq + sigma2_sq + C2)),
                 (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                             (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value, axis=[1, 2, 3])
    return value


def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        filtered_im1 = tf.nn.avg_pool(img1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool(img2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2

    # list to tensor of dim D+1
    mssim = tf.pack(mssim, axis=0)
    mcs = tf.pack(mcs, axis=0)

    value = (tf.reduce_prod(mcs[0:level - 1] ** weight[0:level - 1]) *
             (mssim[level - 1] ** weight[level - 1]))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value