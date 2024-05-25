import tensorflow as tf
import unittest
class test_cuda(unittest.TestCase):
    def test_cuda1(self):
        gpus = tf.config.list_physical_devices('GPU')
        print("Num GPUs Available: ", len(gpus))
        print(gpus)

    def test_cuda2(self):
        is_cuda_available = tf.test.is_built_with_cuda()
        print("Is built with CUDA: ", is_cuda_available)

    def test_use_cuda(self):
        tensor = tf.random.uniform([1000, 1000])

        # 矩阵相乘，一个典型的GPU加速操作
        result = tf.matmul(tensor, tf.transpose(tensor))
        print("Shape of result:", result.shape)

        # 检查结果张量的执行设备
        print("Device:", result.device)

    def test_force(self):
        with tf.device('/GPU:0'):
            # 在GPU上创建并执行操作
            tensor_gpu = tf.random.uniform([1000, 1000])
            result_gpu = tf.matmul(tensor_gpu, tf.transpose(tensor_gpu))
            print(result_gpu.device)
