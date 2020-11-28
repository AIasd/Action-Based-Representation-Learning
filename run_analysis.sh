export SRL_DATASET_PATH=/home/zhongzzy9/Documents/self-driving-car/Action-Based-Representation-Learning
export ACTIONDIR=/home/zhongzzy9/Documents/self-driving-car/Action-Based-Representation-Learning
export PYTHONPATH=$PYTHONPATH:$ACTIONDIR/Carla96ped4/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$ACTIONDIR/Carla96ped4/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:$ACTIONDIR/scenario_runner
export PYTHONPATH=$PYTHONPATH:$ACTIONDIR/carl

python analyze_embeddings.py
