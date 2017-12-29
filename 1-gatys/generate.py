import tensorflow as tf
import numpy as np
import time
import argparse

from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def args_parser():
    parser = argparse.ArgumentParser(description='A Neural Algorithm of Artistic Style')
    parser.add_argument('style_image')
    parser.add_argument('content_image')
    parser.add_argument('--out', default='result.jpg', help='Path to save transferred result', type=str)
    parser.add_argument('--content_size', default=256, help='Content image size', type=int)
    parser.add_argument('--style_size', default=256, help='Style image size', type=int)
    parser.add_argument('--keep_colors', default=False, action='store_true')
    parser.add_argument('--learning_rate', default=10, help='Learning rate', type=int)
    parser.add_argument('--num_iter', default=500, help='Number of iterations', type=int)
    parser.add_argument('--log_iter', default=100, help='Log interval', type=int)
    parser.add_argument('--vgg_weights', default='../vgg16_weights.npz', help='Vgg weights path', type=str)
    parser.add_argument('--content_weight', default=1, help='Content loss weight', type=float)
    parser.add_argument('--style_weight', default=10000, help='Style loss weight', type=float)
    return parser.parse_args()


def vgg16(x, weights):
    # substract imagenet mean
    mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='imagenet_mean')
    x = x - mean

    with tf.variable_scope("vgg16", reuse=tf.AUTO_REUSE):
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.get_variable(initializer=tf.constant(weights["conv1_1_W"]), trainable=False, name='conv1_1_W')
            biases = tf.get_variable(initializer=tf.constant(weights["conv1_1_b"]), trainable=False, name='conv1_1_b')
            conv1_1 = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
            conv1_1 = tf.nn.bias_add(conv1_1, biases)
            conv1_1 = tf.nn.relu(conv1_1, name=scope)

        with tf.name_scope('conv1_2') as scope:
            kernel = tf.get_variable(initializer=tf.constant(weights["conv1_2_W"]), trainable=False, name='conv1_2_W')
            biases = tf.get_variable(initializer=tf.constant(weights["conv1_2_b"]), trainable=False, name='conv1_2_b')
            conv1_2 = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            conv1_2 = tf.nn.bias_add(conv1_2, biases)
            conv1_2 = tf.nn.relu(conv1_2, name=scope)

        pool1 = tf.nn.avg_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                               name='pool1')

        with tf.name_scope('conv2_1') as scope:
            kernel = tf.get_variable(initializer=tf.constant(weights["conv2_1_W"]), trainable=False, name='conv2_1_W')
            biases = tf.get_variable(initializer=tf.constant(weights["conv2_1_b"]), trainable=False, name='conv2_1_b')
            conv2_1 = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            conv2_1 = tf.nn.bias_add(conv2_1, biases)
            conv2_1 = tf.nn.relu(conv2_1, name=scope)

        with tf.name_scope('conv2_2') as scope:
            kernel = tf.get_variable(initializer=tf.constant(weights["conv2_2_W"]), trainable=False, name='conv2_2_W')
            biases = tf.get_variable(initializer=tf.constant(weights["conv2_2_b"]), trainable=False, name='conv2_2_b')
            conv2_2 = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            conv2_2 = tf.nn.bias_add(conv2_2, biases)
            conv2_2 = tf.nn.relu(conv2_2, name=scope)

        pool2 = tf.nn.avg_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                               name='pool2')

        with tf.name_scope('conv3_1') as scope:
            kernel = tf.get_variable(initializer=tf.constant(weights["conv3_1_W"]), trainable=False, name='conv3_1_W')
            biases = tf.get_variable(initializer=tf.constant(weights["conv3_1_b"]), trainable=False, name='conv3_1_b')
            conv3_1 = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            conv3_1 = tf.nn.bias_add(conv3_1, biases)
            conv3_1 = tf.nn.relu(conv3_1, name=scope)

        with tf.name_scope('conv3_2') as scope:
            kernel = tf.get_variable(initializer=tf.constant(weights["conv3_2_W"]), trainable=False, name='conv3_2_W')
            biases = tf.get_variable(initializer=tf.constant(weights["conv3_2_b"]), trainable=False, name='conv3_2_b')
            conv3_2 = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            conv3_2 = tf.nn.bias_add(conv3_2, biases)
            conv3_2 = tf.nn.relu(conv3_2, name=scope)

        with tf.name_scope('conv3_3') as scope:
            kernel = tf.get_variable(initializer=tf.constant(weights["conv3_3_W"]), trainable=False, name='conv3_3_W')
            biases = tf.get_variable(initializer=tf.constant(weights["conv3_3_b"]), trainable=False, name='conv3_3_b')
            conv3_3 = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            conv3_3 = tf.nn.bias_add(conv3_3, biases)
            conv3_3 = tf.nn.relu(conv3_3, name=scope)

        pool3 = tf.nn.avg_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                               name='pool3')

        with tf.name_scope('conv4_1') as scope:
            kernel = tf.get_variable(initializer=tf.constant(weights["conv4_1_W"]), trainable=False, name='conv4_1_W')
            biases = tf.get_variable(initializer=tf.constant(weights["conv4_1_b"]), trainable=False, name='conv4_1_b')
            conv4_1 = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
            conv4_1 = tf.nn.bias_add(conv4_1, biases)
            conv4_1 = tf.nn.relu(conv4_1, name=scope)

        with tf.name_scope('conv4_2') as scope:
            kernel = tf.get_variable(initializer=tf.constant(weights["conv4_2_W"]), trainable=False, name='conv4_2_W')
            biases = tf.get_variable(initializer=tf.constant(weights["conv4_2_b"]), trainable=False, name='conv4_2_b')
            conv4_2 = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            conv4_2 = tf.nn.bias_add(conv4_2, biases)
            conv4_2 = tf.nn.relu(conv4_2, name=scope)

        with tf.name_scope('conv4_3') as scope:
            kernel = tf.get_variable(initializer=tf.constant(weights["conv4_3_W"]), trainable=False, name='conv4_3_W')
            biases = tf.get_variable(initializer=tf.constant(weights["conv4_3_b"]), trainable=False, name='conv4_3_b')
            conv4_3 = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            conv4_3 = tf.nn.bias_add(conv4_3, biases)
            conv4_3 = tf.nn.relu(conv4_3, name=scope)

    return conv1_2, conv2_2, conv3_3, conv4_3


def gram_matrix(x):
    batch_size, w, h, ch = x.shape.as_list()
    x = tf.reshape(x, [batch_size, w * h, ch])
    return tf.matmul(x, x, transpose_a=True) / (ch * w * h)


def loss_fun(target_style_features, target_content_features, transferred_features,
             style_loss_weight, content_loss_weight):
    # relu3_3 as content features  instead of relu4_2 in the original work
    content_loss = 2 * tf.nn.l2_loss(target_content_features[2] - transferred_features[2])

    # relu1_2, relu2_2, relu3_3, relu4_3 as style features
    # instead of relu1_1, relu2_1, ..., relu5_1 in the original work
    style_loss = 0
    for i in range(len(transferred_features)):
        gram_target = gram_matrix(target_style_features[i])
        gram_transferred = gram_matrix(transferred_features[i])
        style_loss = style_loss + 2 * tf.nn.l2_loss(gram_target - gram_transferred)

    return content_loss_weight * content_loss + style_loss_weight * style_loss


def main():
    args = args_parser()

    # load and resize style image and convert it to ndarray
    style_image = tf.keras.preprocessing.image.img_to_array(
        tf.keras.preprocessing.image.load_img(args.style_image,
                                              target_size=(args.style_size, args.style_size)))

    # load and resize content image and convert it to ndarray
    content_image = tf.keras.preprocessing.image.img_to_array(
        tf.keras.preprocessing.image.load_img(args.content_image,
                                              target_size=(args.content_size, args.content_size)))

    # load vgg weights
    vgg_weights = np.load(args.vgg_weights)

    tf.reset_default_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Graph().as_default() as g, tf.Session(config=config) as sess:
        s = sess.run(tf.expand_dims(style_image, axis=0))
        c = sess.run(tf.expand_dims(content_image, axis=0))

        style = tf.placeholder(name="style", dtype=tf.float32, shape=[1, args.style_size, args.style_size, 3])
        content = tf.placeholder(name="content", dtype=tf.float32, shape=[1, args.content_size, args.content_size, 3])

        # the result image to optimize
        transferred = tf.clip_by_value(tf.Variable(initial_value=c, dtype=tf.float32), 0, 255)

        # extract style image's features
        target_style_features = vgg16(style, vgg_weights)

        # extract content image's features
        target_content_features = vgg16(content, vgg_weights)

        # extract the result image's features
        transferred_features = vgg16(transferred, vgg_weights)

        # calculate the loss between transferred image and the original image + style
        loss = loss_fun(target_style_features, target_content_features, transferred_features,
                        args.style_weight, args.content_weight)

        train_op = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss)

        sess.run(tf.global_variables_initializer())

        start = time.time()
        for i in range(args.num_iter):
            it = i + 1

            _, cur_loss = sess.run([train_op, loss], feed_dict={style: s, content: c})

            if it % args.log_iter == 0:
                print("Iteration: [{it}/{num_iter}], loss: {loss}".format(it=it, num_iter=args.num_iter,
                                                                          loss=cur_loss))

        end = time.time()
        print("Finished {num_iter} iteration in {time} seconds".format(num_iter=args.num_iter, time=end - start))
        result = sess.run(tf.squeeze(transferred))

    if args.keep_colors:
        # retain original image color
        def use_original_color(original, result):
            result_hsv = rgb_to_hsv(result)
            orig_hsv = rgb_to_hsv(original)
            oh, os, ov = np.split(orig_hsv, axis=-1, indices_or_sections=3)
            rh, rs, rv = np.split(result_hsv, axis=-1, indices_or_sections=3)
            return hsv_to_rgb(np.concatenate([oh, os, rv], axis=-1))

        final_result = use_original_color(content_image.reshape((args.content_size, args.content_size, 3)), result)
    else:
        final_result = result

    result_image = tf.keras.preprocessing.image.array_to_img(final_result)
    result_image.save(args.out)


main()
