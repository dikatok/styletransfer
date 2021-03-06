import tensorflow as tf
import numpy as np
import time
import argparse

from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def args_parser():
    parser = argparse.ArgumentParser(description='Multi-style Generative Network for Real-time Transfer')
    parser.add_argument('model_dir')
    parser.add_argument('style_image')
    parser.add_argument('content_image')
    parser.add_argument('--out', default='result.jpg', help='Path to save transferred result', type=str)
    parser.add_argument('--keep_colors', default=False, action='store_true')
    return parser.parse_args()


def main():
    args = args_parser()

    style_image = tf.keras.preprocessing.image.img_to_array(
        tf.keras.preprocessing.image.load_img(args.style_image, target_size=(256,256)))

    content_image = tf.keras.preprocessing.image.img_to_array(
        tf.keras.preprocessing.image.load_img(args.content_image))

    tf.reset_default_graph()
    eval_graph = tf.Graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with eval_graph.as_default() as g, tf.Session(config=config, graph=eval_graph) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], args.model_dir)

        c, s = sess.run([tf.expand_dims(content_image, axis=0), tf.expand_dims(style_image, axis=0)])

        inputs = g.get_tensor_by_name("inputs:0")
        style = g.get_tensor_by_name("style:0")
        outputs = g.get_tensor_by_name("outputs:0")

        pnet_out = {}
        for op in tf.get_default_graph().get_operations():
            if op.name.startswith("gammas_out") or op.name.startswith("betas_out"):
                pnet_out[op.name.replace("out", "in")] = g.get_tensor_by_name(op.name + ":0")

        pnet_out_list = sess.run([op for _, op in pnet_out.items()], feed_dict={style: s})

        tnet_in = {inputs: c}
        i = 0
        for op_name in pnet_out.keys():
            tnet_in[g.get_tensor_by_name(op_name.replace("out", "in") + ":0")] = pnet_out_list[i]
            i += 1

        start = time.time()
        result = sess.run(tf.squeeze(outputs), feed_dict=tnet_in)
        end = time.time()
        print("Inference time: {time} seconds".format(time=end-start))

    if args.keep_colors:
        # retain original image color
        def use_original_color(original, result):
            result_hsv = rgb_to_hsv(result)
            orig_hsv = rgb_to_hsv(original)
            oh, os, ov = np.split(orig_hsv, axis=-1, indices_or_sections=3)
            rh, rs, rv = np.split(result_hsv, axis=-1, indices_or_sections=3)
            return hsv_to_rgb(np.concatenate([oh, os, rv], axis=-1))

        final_result = use_original_color(content_image.reshape((0, 0, 3)), result)
    else:
        final_result = result

    result_image = tf.keras.preprocessing.image.array_to_img(final_result)
    result_image.save(args.out)


main()