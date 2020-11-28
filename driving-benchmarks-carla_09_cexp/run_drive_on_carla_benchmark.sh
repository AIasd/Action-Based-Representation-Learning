export SRL_DATASET_PATH=/home/zhongzzy9/Documents/self-driving-car/Action-Based-Representation-Learning
export ACTIONDIR=/home/zhongzzy9/Documents/self-driving-car/Action-Based-Representation-Learning

export PYTHONPATH=$ACTIONDIR:$ACTIONDIR/cad:$ACTIONDIR/Carla96ped4/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg/:$ACTIONDIR/Carla96ped4/PythonAPI/carla:$ACTIONDIR/scenario_runner

python3 benchmark_runner.py -b NoCrash -a $ACTIONDIR/drive/AffordancesAgent.py -d carlaped -c $ACTIONDIR/_logs/EXP/BC_smallDataset_seed1_encoder_frozen_1FC_smallDataset_s1_1000/config.json --gpu 0


# python3 benchmark_runner.py -b NoCrash -a $ACTIONDIR/drive/AffordancesAgent.py -d carlaped -c $ACTIONDIR/_logs/EXP/BC_im_50Hours_seed1_encoder_finetuning_3FC_5Hours_s1_100000/config.json --gpu 0
