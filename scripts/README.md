# Scripts

run with tensorflow 2.15 : 
conda activate tf215

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:\$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_nvcc
export TF_CPP_MIN_LOG_LEVEL=2


python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU')); tf.constant(1.0)**2"
If GPUs: [], check LD_LIBRARY_PATH.
If JIT compilation failed, check XLA_FLAGS.

