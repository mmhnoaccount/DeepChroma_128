import tfrecord
import numpy as np
import tensorflow as tf

# writer = tfrecord.TFRecordWriter("E:/File/package_by_mmh/train_set_128/train.tfrecord")
writer = tf.compat.v1.python_io.TFRecordWriter("E:/File/package_by_mmh/train_set_128/A_test.tfrecord")
'''
writer.write({'length': (3, 'int'), 'label': (1, 'int')},
             {'tokens': ([[0, 0, 1], [0, 1, 0], [1, 0, 0]], 'float'), 'seq_labels': ([0, 1, 1], 'float')}) # changed
writer.write({'length': (3, 'int'), 'label': (1, 'int')},
             {'tokens': ([[0, 0, 1], [1, 0, 0]], 'float'), 'seq_labels': ([0, 1], 'float')})
'''
length = 3
width = 3
img_raw = np.array([[2.2, 6.7], [4.7, 5.2]], dtype='float')
img_raw = img_raw.tobytes()
example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                        # "width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width]))
                        # 'length': tf.train.Feature(int64_list=tf.train.Int64List(value=[length])),
                        # 'img_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[3]))
                    }
                )

            )
writer.write(example.SerializeToString())

writer.close()


loader = tfrecord.tfrecord_loader("E:/File/package_by_mmh/train_set_128/A_test.tfrecord", None, {
    'img_raw': 'byte'})

for record in loader:
    aaa = np.frombuffer(record['img_raw']).reshape(2, 2)
    #print('img_name:{}'.format(record['img_raw']))
    print(aaa)
