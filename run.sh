export SRL_DATASET_PATH=/home/zhongzzy9/Documents/self-driving-car/Action-Based-Representation-Learning
export ACTIONDIR=/home/zhongzzy9/Documents/self-driving-car/Action-Based-Representation-Learning
export PYTHONPATH=$PYTHONPATH:$ACTIONDIR/Carla96ped4/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$ACTIONDIR/Carla96ped4/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:$ACTIONDIR/scenario_runner
export PYTHONPATH=$PYTHONPATH:$ACTIONDIR/carl


# python3 main.py --single-process train_encoder --gpus 0 --encoder-folder ENCODER --encoder-exp BC_smallDataset_seed1

# python3 main.py --single-process train --gpus 0 --encoder-folder ENCODER --encoder-exp BC_smallDataset_seed1 --encoder-checkpoint 1000 -f EXP -e BC_smallDataset_seed1_encoder_frozen_1FC_smallDataset_s1


python3 main.py --single-process validation --gpus 0 --encoder-folder ENCODER --encoder-exp BC_smallDataset_seed1 --encoder-checkpoint 1000 -f EXP -e BC_smallDataset_seed1_encoder_frozen_1FC_smallDataset_s1 -vj $ACTIONDIR/carl/database/CoRL2020/small_dataset.json


# docker image build -f Carla96ped4/Dockerfile -t carlaped Carla96ped4/
# docker image build -f dockerfile -t carlaped Carla96ped4/

# python carl/multi_gpu_data_collection.py -j '/home/zhongzzy9/Documents/self-driving-car/Action-Based-Representation-Learning/carl/database/CoRL2020/small_dataset.json' -ct carlaped -ge 1 2 3 4 5 6 7 8 9
