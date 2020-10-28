import tensorflow.compat.v1 as tf
import cv2
import numpy as np


def load_model(pb_path):
    tf.reset_default_graph()
    sess = tf.Session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.Session(config=config)

    with tf.gfile.GFile(pb_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    sess.graph.as_default()
    tf.import_graph_def(graph_def)
    return sess


def preprocess(image):
    image_rev = np.flip(image, axis=1)
    return np.stack([image, image_rev], axis=0)


def postprocess(out, input_shape=(384, 384)):
    """
    Create a grayscale human segmentation map

        LIP Classes
        -------------
         0 | Background
         1 | Hat
         2 | Hair
         3 | Glove
         4 | Sunglass
         5 | UpperClothes
         6 | Dress
         7 | Coat
         8 | Socks
         9 | Pants
        10 |  Jumpsuits
        11 |  Scarf
        12 |  Skirt
        13 |  Face
        14 |  LeftArm
        15 |  RightArm
        16 |  LeftLeg
        17 |  RightLeg
        18 |  LeftShoe
        19 |  RightShoe
    """
    head_output, tail_output = tf.unstack(out, num=2, axis=0)
    tail_list = tf.unstack(tail_output, num=20, axis=2)
    tail_list_rev = [None] * 20
    tail_list_rev = [tail_list[i] for i in range(14)]

    tail_list_rev.extend(
        [
            tail_list[15],
            tail_list[14],
            tail_list[17],
            tail_list[16],
            tail_list[19],
            tail_list[18],
        ]
    )
    tail_output_rev = tf.stack(tail_list_rev, axis=2)
    tail_output_rev = tf.reverse(tail_output_rev, tf.stack([1]))
    raw_output_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
    raw_output_all = tf.expand_dims(raw_output_all, dim=0)
    raw_output_all = tf.argmax(raw_output_all, dimension=3)
    pred_all = tf.expand_dims(raw_output_all, dim=3).numpy()  # Create 4-d tensor.
    # raw_output_all = raw_output_all.transpose(1, 2, 0)
    print("output: ", pred_all.shape)
    print("raw output: ", raw_output_all.shape)
    return pred_all[0, :, :, 0]


def colorize_segmentation(gray_image):
    """
    Colorize image according to labels
    """
    height, width = gray_image.shape[:2]

    colors = [(0, 0, 0), (128, 0, 0), (255, 0, 0), (0, 85, 0), (170, 0, 51), (255, 85, 0),
              (0, 0, 85), (0, 119, 221), (85, 85, 0), (0, 85, 85), (85, 51, 0), (52, 86, 128),
              (0, 128, 0), (0, 0, 255), (51, 170, 221), (0, 255, 255),(85, 255, 170),
              (170, 255, 85), (255, 255, 0), (255, 170, 0)]
    
    segm = np.stack([colors[idx] for idx in gray_image.flatten()])
    segm = segm.reshape(height, width, 3).astype(np.uint8)
    segm = cv2.cvtColor(segm, cv2.COLOR_BGR2RGB)
    return segm


def predict(sess, image):
    inputs_ = preprocess(image)
    input_h, input_w = inputs_.shape[2:]
    
    outputs_ = sess.run('import/Mean_3:0', feed_dict={'import/input:0': inputs_})
    print(outputs_.shape)
    grayscale_out = postprocess(outputs_, (input_w, input_h))
    return grayscale_out


if __name__ == "__main__":
    img = cv2.imread('../data/person.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # sess = load_model('../lip_jppnet_384.pb')
    sess = load_model('../LIP_JPPNet.pb')
    segmap = predict(sess, img)
    segmap_color = colorize_segmentation(segmap)
    cv2.imwrite("../segmap.jpg", segmap)
    cv2.imwrite("../segmap_color.jpg", segmap_color)
