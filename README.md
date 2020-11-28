# Action-Based Representation Learning for Autonomous Driving

-------------------------------------------------------------
This repository is forked from code associated with the paper: Action-Based Representation Learning for Autonomous Driving

 <img src="driving_clip.gif" height="350">


-------------------------------------------------------------
### Experiments Summary

The processes can be defined as four types:

 * Train an encoder model (Behaviour Cloning (BC), Inverse, Forward, ST-DIM)
 * Train a MLP for affordances outputs. The pre-trained encoder model will be used.
 * Validation on affordances prediction.
 * Actual drive using controller tuned with affordances prediction.


-------------------------------------------------------------
### Setting Environments & Getting Datasets

1. Download the dataset `customized`.

2. change path to your dataset folder with `SRL_DATASET_PATH` in `run_customized.sh` and `run_analysis.sh`:

3. Download the repository
```
        git clone https://github.com/AIasd/Action-Based-Representation-Learning.git
```
4. change path to your downloaded repository with `ACTIONDIR` in `run_customized.sh` and `run_analysis.sh`:


5. Download the CARLA version we used with this [link](https://drive.google.com/file/d/1m4J2yJqL7QcCfaxvMh8erLzdGEyFC5mg/view?usp=sharing), and put it inside your downloaded repository folder

6. change the absolute path `file_dir` in `carl/cexp/env/environment.py` to your local folder path.
-------------------------------------------------------------
### Training Encoder

1. Define configuration files for training. Refer to [files](https://github.com/AIasd/Action-Based-Representation-Learning/tree/master/configs/ENCODER) in configs folder

2. Run the `run_customized.sh` file with "train_encoder" process by only commenting in the corresponding line: `python3 main.py --single-process train_encoder --gpus 0 --encoder-folder ENCODER --encoder-exp customized` where `--single-process` defines the process type, `--gpus` defines the gpu to be used, `--encoder-folder` is the experiment folder you defined in [config folder](https://github.com/AIasd/Action-Based-Representation-Learning/tree/master/configs), and `--encoder-exp` is the experiment you defined inside the experiment folder.

-------------------------------------------------------------
### Training MLP for affordances

1. Define configuration files for training. Refer to [files](https://github.com/AIasd/Action-Based-Representation-Learning/tree/master/configs/EXP) in configs folder

2. Run the `run_customized.sh` file with "train_encoder" process by only commenting in the corresponding line: `python3 main.py --single-process train --gpus 0 --encoder-folder ENCODER --encoder-exp customized --encoder-checkpoint 30000 -f EXP -e customized`

-------------------------------------------------------------
### Validate on affordances prediction

1. Run the `run_customized.sh` file with "train_encoder" process by only commenting in the corresponding line: `python3 main.py --single-process validation --gpus 0 --encoder-folder ENCODER --encoder-exp customized --encoder-checkpoint 30000 -f EXP -e customized -vj $ACTIONDIR/carl/database/CoRL2020/customized.json`

-------------------------------------------------------------


### Video
[video](https://www.youtube.com/watch?v=fFywCMlLbyE)

### Reference:
 * Yi Xiao, Felipe Codevilla, Christopher Pal, Antonio M. Lopez, [Action-Based Representation Learning for Autonomous Driving](https://arxiv.org/abs/2008.09417).

        @article{Xiao2020ActionBasedRL,
        title={Action-Based Representation Learning for Autonomous Driving},
        author={Y. Xiao and Felipe Codevilla and C. Pal and Antonio M. L{\'o}pez},
        journal={ArXiv},
        year={2020},
        volume={abs/2008.09417}
        }
 * [Coiltraine](https://github.com/felipecode/coiltraine), which can be used to easily train and manage the trainings of imitation learning networks jointly with evaluations on the CARLA simulator.
 * [Cexp](https://github.com/felipecode/cexp), which is a interface to the CARLA simulator and the scenario runner to produce fully usable environments.
