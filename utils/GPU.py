#Set the percentage of cuda cores (1 = 100%)

import tensorflow as T

fraction = 0.90

def set_gpu_fraction(sess=None, gpu_fraction=fraction):
    """Set the GPU memory fraction for the application.

    Parameters
    ----------
    sess : a session instance of TensorFlow
        TensorFlow session
    gpu_fraction : a float
        Fraction of GPU memory, (0 ~ 1]

    References
    ----------
    - `TensorFlow using GPU <https://www.tensorflow.org/versions/r0.9/how_tos/using_gpu/index.html>`_
    """
    print("  tensorlayer: GPU MEM Fraction %f" % gpu_fraction)
    gpu_options = T.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    sess = T.compat.v1.Session(config = T.compat.v1.ConfigProto(gpu_options = gpu_options))
    return sess 


set_gpu_fraction(None,fraction)
print("Num GPUs Available: ", len(T.config.list_physical_devices('GPU')))