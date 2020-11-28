


import os
import sys
import random
import time
import traceback
import torch
import torch.optim as optim

from configs import g_conf, set_type_of_process, merge_with_yaml
from network import CoILModel, Loss, adjust_learning_rate_auto, EncoderModel
from input import CoILDataset, Augmenter, select_balancing_strategy
from logger import coil_logger
from coilutils.checkpoint_schedule import is_ready_to_save, get_latest_saved_checkpoint, check_loss_validation_stopped
import numpy as np
import pandas as pd
import json

def seed_everything(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


def gather_data():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'


    exp_batch = 'EXP'
    exp_alias = 'customized'
    number_of_workers = 12
    encoder_params = {'encoder_checkpoint': 10000,
                      'encoder_folder': 'ENCODER',
                      'encoder_exp': 'customized'}



    try:
        merge_with_yaml(os.path.join('configs', exp_batch, exp_alias + '.yaml'), encoder_params)
        g_conf.PROCESS_NAME = 'analyze_embeddings'


        # We can save preload dataset depends on the json file name, then no need to load dataset for each time with the same dataset
        if len(g_conf.EXPERIENCE_FILE) == 1:
            json_file_name = str(g_conf.EXPERIENCE_FILE[0]).split('/')[-1].split('.')[-2]
        else:
            json_file_name = str(g_conf.EXPERIENCE_FILE[0]).split('/')[-1].split('.')[-2] + '_' + str(g_conf.EXPERIENCE_FILE[1]).split('/')[-1].split('.')[-2]
        dataset = CoILDataset(transform=None,
                              preload_name=g_conf.PROCESS_NAME + '_' + json_file_name + '_' + g_conf.DATA_USED)

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=g_conf.BATCH_SIZE, sampler=None, num_workers=12, pin_memory=True)


        print ("Loaded Training dataset")

        # we use the pre-trained encoder model to extract bottleneck Z and train the E-t-E model
        iteration = 0
        if g_conf.MODEL_TYPE in ['separate-affordances']:
            encoder_model = EncoderModel(g_conf.ENCODER_MODEL_TYPE, g_conf.ENCODER_MODEL_CONFIGURATION)
            encoder_model.cuda()
            encoder_model.eval()
            # To freeze the pre-trained encoder model
            if g_conf.FREEZE_ENCODER:
                for param_ in encoder_model.parameters():
                    param_.requires_grad = False

            encoder_checkpoint = torch.load(
                os.path.join('_logs', encoder_params['encoder_folder'], encoder_params['encoder_exp'], 'checkpoints',
                             str(encoder_params['encoder_checkpoint']) + '.pth'))
            print("Encoder model ", str(encoder_params['encoder_checkpoint']), "loaded from ",
                  os.path.join('_logs', encoder_params['encoder_folder'], encoder_params['encoder_exp'], 'checkpoints'))
            encoder_model.load_state_dict(encoder_checkpoint['state_dict'])

        print('g_conf.ENCODER_MODEL_TYPE', g_conf.ENCODER_MODEL_TYPE)



        behaviors = []
        route_folder = 'customized'
        route_files = os.listdir(route_folder)
        count = 0

        kept_inds = []
        total_length = 0
        for route_file in sorted(route_files):
            print('-'*100, route_file)
            route_path = os.path.join(route_folder, route_file)
            if os.path.isdir(route_path):
                measurements_path = route_path + '/' + 'driving_log.csv'

                import pandas as pd
                df = pd.read_csv(measurements_path, delimiter=',')





                # remove cases which has ego_linear_speed < 0.1 or other_actor_linear_speed < 0
                ego_fault = True

                events_path = route_path + '/' + 'events.txt'
                with open(events_path) as json_file:
                    events = json.load(json_file)
                infractions = events['_checkpoint']['records'][0]['infractions']
                import re
                infraction_types = ['collisions_layout', 'collisions_pedestrian', 'collisions_vehicle', 'red_light', 'on_sidewalk', 'outside_lane_infraction', 'wrong_lane', 'off_road']
                for infraction_type in infraction_types:
                    for infraction in infractions[infraction_type]:
                        if 'collisions' in infraction_type:
                            typ = re.search('.*with type=(.*) and id.*', infraction).group(1)
                            if 'walker' not in typ:
                                ego_fault = False
                            else:
                                loc = re.search('.*x=(.*), y=(.*), z=(.*), ego_linear_speed=(.*), other_actor_linear_speed=(.*)\)', infraction)
                                if loc:
                                    ego_linear_speed = float(loc.group(4))
                                    other_actor_linear_speed = float(loc.group(5))
                                    if ego_linear_speed < 0.1 or other_actor_linear_speed < 0:
                                        ego_fault = False
                        else:
                            loc = re.search('.*x=(.*), y=(.*), z=(.*)', infraction)
                            if loc:
                                ego_fault = False

                cur_behaviors = df[['behaviors']].to_numpy()
                if ego_fault:
                    behaviors.append(cur_behaviors)
                    kept_inds.append(np.array(range(total_length, total_length+cur_behaviors.shape[0])))
                    count += 1
                total_length += cur_behaviors.shape[0]

        kept_inds = np.concatenate(kept_inds)

        print(f'{count}/{len(route_files)} are used')




        misbehavior_labels = np.concatenate(behaviors).squeeze()
        print(misbehavior_labels.shape, len(dataset))
        print(np.sum(misbehavior_labels==0), np.sum(misbehavior_labels==1))




        num_of_iterations = len(dataset) // g_conf.BATCH_SIZE
        image_embeddings = []
        for data in data_loader:
            inputs_data = torch.squeeze(data['rgb'].cuda())

            # separate_affordance
            e, inter = encoder_model.forward_encoder(inputs_data,
                                                      dataset.extract_inputs(data).cuda(),torch.squeeze(dataset.extract_commands(data).cuda()))
            # print(len(e), e[0].shape)
            image_embeddings.append(e.cpu())
            iteration += 1
            if iteration == num_of_iterations:
                break
        image_embeddings = np.concatenate(image_embeddings, axis=0)
        image_embeddings = image_embeddings[kept_inds]

        print(image_embeddings.shape, misbehavior_labels.shape)
        np.savez('misbehavior/embeddings_and_misbehavior_labels', image_embeddings=image_embeddings, misbehavior_labels=misbehavior_labels[:len(image_embeddings)])

        coil_logger.add_message('Finished', {})

    except KeyboardInterrupt:
        coil_logger.add_message('Error', {'Message': 'Killed By User'})

    except RuntimeError as e:

        coil_logger.add_message('Error', {'Message': str(e)})

    except:
        traceback.print_exc()
        coil_logger.add_message('Error', {'Message': 'Something Happened'})



def classify():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_moons, make_circles, make_classification
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.dummy import DummyClassifier

    def subsample(X, y):
        inds = np.arange(y.shape[0])

        chosen_inds_0 = y==0
        chosen_inds_1 = y==1
        X_0 = X[chosen_inds_0][::120]
        y_0 = y[chosen_inds_0][::120]
        inds_0 = inds[chosen_inds_0][::120]
        X_1 = X[chosen_inds_1]
        y_1 = y[chosen_inds_1]
        inds_1 = inds[chosen_inds_1]


        # X_1 = np.repeat(X_1, 8, axis=0)
        # y_1 = np.repeat(y_1, 8)

        print('X_0:', X_0.shape, 'X_1:', X_1.shape, 'y_0:', y_0.shape, 'y_1:', y_1.shape)
        X = np.concatenate([X_0, X_1], axis=0)
        y = np.concatenate([y_0, y_1], axis=0)
        inds = np.concatenate([inds_0, inds_1], axis=0)

        perm = np.random.permutation(len(X))
        X = X[perm]
        y = y[perm]
        inds = inds[perm]
        return X, y, inds

    data = np.load('misbehavior/embeddings_and_misbehavior_labels.npz')
    image_embeddings = data['image_embeddings']
    misbehavior_labels = data['misbehavior_labels']
    X, y = image_embeddings, misbehavior_labels


    # -------------------------------------------------------------------------
    # a simple GT pedestrian distance baseline
    total_data_dir = 'customized'
    closest_pedestrian_distance = []

    for sub_dir in os.listdir(total_data_dir):
        data_dir = os.path.join(total_data_dir, sub_dir)
        if os.path.isdir(data_dir):
            measurements_folder = os.path.join(data_dir, '0_NPC', '0')
            for measurement_name in os.listdir(measurements_folder):
                if measurement_name[:13] == 'measurements_':
                    measurement_path = os.path.join(measurements_folder, measurement_name)
                    with open(measurement_path, 'r') as f_in:
                        measurement = json.load(f_in)
                        closest_pedestrian_distance.append(measurement['closest_pedestrian_distance'])

    closest_pedestrian_distance = np.array(closest_pedestrian_distance)
    print(closest_pedestrian_distance.shape)
    closest_pedestrian_distance = closest_pedestrian_distance[:y.shape[0]]
    # -------------------------------------------------------------------------






    X = StandardScaler().fit_transform(X)

    print(X.shape, y.shape)
    cutoff = int(X.shape[0] * 0.7)
    X_train, X_test = X[:cutoff], X[cutoff:]
    y_train, y_test = y[:cutoff], y[cutoff:]



    X_train, y_train, inds_train = subsample(X_train, y_train)
    X_test, y_test, inds_test = subsample(X_test, y_test)
    # print(inds_train[:10], inds_test[:10])
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


    pred_cutoff = int(np.mean(y_train == 1) * y_test.shape[0])


    closest_pedestrian_distance_test = closest_pedestrian_distance[inds_test]
    closest_pedestrian_distance_test_inds = [(d, i) for i, d in enumerate(closest_pedestrian_distance_test)]



    y_ped_dist_pred = np.zeros(y_test.shape[0])
    for (d, i) in sorted(closest_pedestrian_distance_test_inds)[:pred_cutoff]:
        y_ped_dist_pred[i] = 1

    ind_0 = y_test==0
    ind_1 = y_test==1
    print(f'using GT ped distance, overall accuracy, {np.mean(y_ped_dist_pred==y_test):.3f}; 0 accuracy: {np.mean(y_ped_dist_pred[ind_0]==y_test[ind_0]):.3f}; 1 accuracy: {np.mean(y_ped_dist_pred[ind_1]==y_test[ind_1]):.3f}')




    # X_train, X_test, y_train, y_test = \
    #     train_test_split(X, y, test_size=.3, random_state=42)


    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Random Forest", "Neural Net", "AdaBoost", "Random"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(), DummyClassifier(strategy='stratified')
        ]


    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        y_pred = clf.predict(X_test)
        print(f'{name}, overall accuracy: {np.mean(y_pred==y_test):.3f}; 0 accuracy: {np.mean(y_pred[ind_0]==y_test[ind_0]):.3f}; 1 accuracy: {np.mean(y_pred[ind_1]==y_test[ind_1]):.3f}')




if __name__ == '__main__':
    seed_everything(999999)
    # gather_data()
    classify()
