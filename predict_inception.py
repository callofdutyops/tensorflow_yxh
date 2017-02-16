import numpy as np
import tensorflow as tf
import os

imagePath = 'replace here'
modelFullPath = '/path to graph/trained_graph.pb'
labelsFullPath = '/path to lable/trained_labels.txt'

def get_images_paths():
    os.chdir(imagePath)
    image_file_paths = []
    image_paths = os.listdir(imagePath)
    image_paths = [os.path.abspath(dir) for dir in image_paths]
    for image_path in image_paths:
        os.chdir(image_path)
        image_file_paths += [os.path.abspath(p) for p in os.listdir(image_path)]
    return image_file_paths

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image():
    answer = []
    right = 0

    #if not tf.gfile.Exists(imagePath):
    #    tf.logging.fatal('File does not exist %s', imagePath)
    #    return answer

    image_datas = [tf.gfile.FastGFile(i, 'rb').read() for i in get_images_paths()]
    image_labels = [str(l).split('/')[-1].split('_')[0].lower() for l in get_images_paths()]

    # Creates graph from saved GraphDef.
    create_graph()

    with tf.Session() as sess:

        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        for index, image_data in enumerate(image_datas):
            predictions = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)

            top_k = predictions.argsort()[-5:][::-1]  # Getting top 5 predictions
            f = open(labelsFullPath, 'rb')
            lines = f.readlines()
            # labels = [str(w).replace("\n", "") for w in lines]
            labels = [str(w) for w in lines]
            for node_id in top_k:
                human_string = labels[node_id]
                score = predictions[node_id]
                print('%s (score = %.5f)' % (human_string, score))

            if image_labels[index] in labels[top_k[0]]:
                right += 1
            acc = right / len(image_labels)
            print('acc = %.5f' % (acc))
        # Because our train acc is 100% and there are 70 test image.
        final_acc = (351 * acc - 281) / 70
        print('final_acc = %.5f' % (final_acc))


if __name__ == '__main__':
    run_inference_on_image()
