export OPENCV_IO_ENABLE_OPENEXR=1
NPROC_PER_NODE=8
WORLD_SIZE=1
RANK=0
MASTER_ADDR=localhost
MASTER_PORT=29400

mkdir -p logs

torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
         train.py --batch_size 16 --dataset oo3d9dsingle --data_path ov9d --data_name oo3d9dsingle --data_train train --data_val test/all \
         --num_filters 32 32 32 --deconv_kernels 2 2 2\
         --save_model --layer_decay 0.9 --log_dir logs \
         --scale_size 480 --epochs 25 --auto_resume --dino 