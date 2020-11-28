export SRL_DATASET_PATH=/home/zhongzzy9/Documents/self-driving-car/Action-Based-Representation-Learning
export ACTIONDIR=/home/zhongzzy9/Documents/self-driving-car/Action-Based-Representation-Learning
export PYTHONPATH=$PYTHONPATH:$ACTIONDIR/Carla96ped4/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$ACTIONDIR/Carla96ped4/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:$ACTIONDIR/scenario_runner
export PYTHONPATH=$PYTHONPATH:$ACTIONDIR/carl

python3 main.py --single-process train_encoder --gpus 0 --encoder-folder ENCODER --encoder-exp customized


# python3 main.py --single-process train --gpus 0 --encoder-folder ENCODER --encoder-exp customized --encoder-checkpoint 30000 -f EXP -e customized


# python3 main.py --single-process validation --gpus 0 --encoder-folder ENCODER --encoder-exp customized --encoder-checkpoint 30000 -f EXP -e customized -vj $ACTIONDIR/carl/database/CoRL2020/customized.json


# docker image build -f Carla96ped4/Dockerfile -t carlaped Carla96ped4/
# docker image build -f dockerfile -t carlaped Carla96ped4/
