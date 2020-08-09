import os
import logging
import json
import sys
import importlib
import shutil
import numpy as np
import traceback
from tqdm import tqdm

from cexp.driving_batch import DrivingBatch


# TODO ADD the posibility to configure what goes in and what goes out ( OUput format)
###


def parse_results_summary(summary):

    result_dictionary = {
        'episodes_completion': summary['score_route'],
        'result': float(summary['result'] == 'SUCCESS'),
        'infractions_score': summary['score_penalty'],
        'number_red_lights': summary['number_red_lights'],
        'total_number_traffic_lights': summary['total_number_traffic_lights']
    }

    return result_dictionary


def read_benchmark_summary(benchmark_csv):
    """
        Make a dict of the benchmark csv were the keys are the environment names

    :param benchmark_csv:
    :return:
    """

    # If the file does not exist, return None,None, to point out that data is missing
    if not os.path.exists(benchmark_csv):
        logging.debug(" Summary Not Produced, No CSV File")
        return None, None

    f = open(benchmark_csv, "r")
    header = f.readline()
    header = header.split(',')
    header[-1] = header[-1][:-2]
    f.close()

    data_matrix = np.loadtxt(open(benchmark_csv, "rb"), delimiter=",", skiprows=1)
    control_results_dic = {}
    count = 0

    if len(data_matrix) == 0:
        logging.debug(" Summary Not Produced, Matrix Empty")
        return None, None
    if len(data_matrix.shape) == 1:
        data_matrix = np.expand_dims(data_matrix, axis=0)

    for env_name in data_matrix[:, 0]:

        control_results_dic.update({env_name: data_matrix[count, 1:]})
        count += 1

    return control_results_dic, header


def read_benchmark_summary_metric(benchmark_csv):
    """
        Make a dict of the benchmark csv were the keys are the environment names

    :param benchmark_csv:
    :return:
    """

    f = open(benchmark_csv, "rU")
    header = f.readline()
    header = header.split(',')
    header[-1] = header[-1][:-2]
    f.close()

    data_matrix = np.loadtxt(benchmark_csv, delimiter=",", skiprows=1)
    summary_dict = {}

    if len(data_matrix) == 0:
        return None

    if len(data_matrix.shape) == 1:
        data_matrix = np.expand_dims(data_matrix, axis=0)

    count = 0
    for _ in header:
        summary_dict.update({header[count]: data_matrix[:, count]})
        count += 1

    return summary_dict


def check_benchmarked_environments(json_filename, agent_checkpoint_name):

    """ return a dict with each environment that has a vector of dicts of results

        The len of each environment is the number of times this environment has been benchmarked.
     """

    benchmarked_environments = {}

    with open(json_filename, 'r') as f:
        json_file = json.loads(f.read())

    if not os.path.exists(os.path.join(os.environ["SRL_DATASET_PATH"], json_file['package_name'])):
        return {}  # return empty dictionary no case was benchmarked

    for env_name in json_file['envs'].keys():
        path = os.path.join(os.environ["SRL_DATASET_PATH"],  json_file['package_name'], env_name,
                            agent_checkpoint_name + '_benchmark_summary.csv')
        if os.path.exists(path):
            env_summary, _ = read_benchmark_summary(path)
            benchmarked_environments.update({env_name: env_summary})

    return benchmarked_environments

def check_benchmark_finished(json_filename, agent_checkpoint_name):

    with open(json_filename, 'r') as f:
        json_file = json.loads(f.read())

    if not os.path.exists(os.path.join(os.environ["SRL_DATASET_PATH"], json_file['package_name'])):
        print (" PACKAGE DOES NOT EXIST")
        return False  # return empty dictionary no case was benchmarked

    for env_name in json_file['envs'].keys():
        path = os.path.join(os.environ["SRL_DATASET_PATH"],  json_file['package_name'], env_name,
                            agent_checkpoint_name + '_benchmark_summary.csv')
        if not os.path.exists(path):
            print (" PATH ", path, "does not exist")
            return False

    return True


def check_benchmarked_episodes_metric(json_filename, agent_checkpoint_name):

    """ return a dict with each metric from the header and

        The len of each environment is the number of times this environment has been benchmarked.
     """

    benchmarked_metric_dict = {}

    with open(json_filename, 'r') as f:
        json_file = json.loads(f.read())

    if not os.path.exists(os.path.join(os.environ["SRL_DATASET_PATH"], json_file['package_name'])):
        return {}  # return empty dictionary no case was benchmarked

    for env_name in json_file['envs'].keys():
        path = os.path.join(os.environ["SRL_DATASET_PATH"],  json_file['package_name'], env_name,
                            agent_checkpoint_name + '_benchmark_summary.csv')

        if os.path.exists(path):
            benchmark_env_results, header = read_benchmark_summary(path)

            if not benchmarked_metric_dict:
                # This is the first iteration, we use it to take the header.
                for key in header[1:]:
                    benchmarked_metric_dict.update({key:[]})

            for rep_key in benchmark_env_results.keys():
                for info, key in zip(benchmark_env_results[rep_key], header[1:]):
                    benchmarked_metric_dict[key].append(info)

    return benchmarked_metric_dict


def summarize_benchmark(summary_data):


    final_dictionary = {}
    # we just get the headers for the final dictionary
    for key in summary_data.keys():
        # The csv reading bug.
        if key == 'total_number_traffic_lights' or key == 'number_red_lights' or\
                key == 'total_number_traffic_light' or key == 'number_red_light':
            continue
        final_dictionary.update({key: 0})

    for metric in summary_data.keys():
        if metric == 'total_number_traffic_lights' or metric == 'number_red_lights' or\
            metric == 'total_number_traffic_light' or metric == 'number_red_light':
            continue
        try:
            final_dictionary[metric] = sum(summary_data[metric]) / len(summary_data[metric])
        except KeyError:  # To overcome the bug on reading files csv
            final_dictionary[metric] = sum(summary_data[metric[:-1]]) / len(summary_data[metric])


    # Weird things for the CSV BUG.
    try:
        final_dictionary.update({'average_percentage_red_lights':
                                     sum(summary_data['number_red_lights'])/
                                         sum(summary_data['total_number_traffic_lights'])
                                 })
    except:
        try:
            final_dictionary.update({'average_percentage_red_lights':
                                         sum(summary_data['number_red_light']) /
                                         sum(summary_data['total_number_traffic_lights'])
                                     })
        except:
            final_dictionary.update({'average_percentage_red_lights':
                                         sum(summary_data['number_red_lights']) /
                                         sum(summary_data['total_number_traffic_light'])
                                     })

    return final_dictionary


def write_summary_csv(out_filename, agent_checkpoint_name, summary_data):

    """
        If produce a csv output that summarizes a benchmark
    """

    fixed_metrics_list = ['episodes_completion',
                           'result',
                           'infractions_score',
                           'average_percentage_red_lights']
    print ("Writting summary")
    print (out_filename)
    #  We check if the csv already exists. TODO check correctness
    # TODO we should remove if it is not succesful written
    # Now we finally make the summary from all the files
    summary_dict = summarize_benchmark(summary_data)

    if not os.path.exists(out_filename):

        csv_outfile = open(out_filename, 'w')
        csv_outfile.write("%s" % 'step')
        for metric in fixed_metrics_list:
            csv_outfile.write(",%s" % metric)

        csv_outfile.write("\n")
        csv_outfile.close()

    csv_outfile = open(out_filename, 'a')
    csv_outfile.write("%s" % (agent_checkpoint_name))

    # TODO AVOID GETTING HERE IF IT FAILS
    for metric in fixed_metrics_list:
        csv_outfile.write(",%f" % (summary_dict[metric]))

    csv_outfile.write("\n")

    csv_outfile.close()



def add_summary(environment_name, summary, json_filename, agent_checkpoint_name):
    """
    Add summary file, if it exist writte another repetition.
    :param environment_name:
    :param summary:
    :param json_filename:
    :param agent_checkpoint_name:
    :return:
    """
    # The rep is now zero, but if the benchmark already started we change that
    repetition_number = 0

    with open(json_filename, 'r') as f:
        json_file = json.loads(f.read())
    # if it doesnt exist we add the file, this is how we are writting.
    filename = os.path.join(os.environ["SRL_DATASET_PATH"], json_file['package_name'],
                            environment_name, agent_checkpoint_name + '_benchmark_summary.csv')
    set_of_metrics = ['episodes_completion', 'result', 'infractions_score',
                      'number_red_lights', 'total_number_traffic_lights']

    if not os.path.exists(filename):
        csv_outfile = open(filename, 'w')
        csv_outfile.write("%s,%s,%s,%s,%s,%s\n"
                          % ('rep', 'episodes_completion', 'result', 'infractions_score',
                             'number_red_lights', 'total_number_traffic_lights'))

        csv_outfile.close()

    else:
        # Check the summary to get the repetition number
        summary_exps = check_benchmarked_environments(json_filename, agent_checkpoint_name)

        env_experiments = summary_exps[environment_name]

        repetition_number = len(env_experiments.keys())

    logging.debug("added summary for " + agent_checkpoint_name + '_benchmark_summary.csv')
    # parse the summary for this episode
    results = parse_results_summary(summary)

    csv_outfile = open(filename, 'a')
    csv_outfile.write("%f" % float(repetition_number) )
    for metric_result in set_of_metrics:
        csv_outfile.write(",%f" % results[metric_result])

    csv_outfile.write("\n")
    csv_outfile.close()


def benchmark_env_loop(renv, agent, save_trajectories=False, draw_relative_angle_error = False):

    sensors_dict = agent.get_sensors_dict()
    renv.set_sensors(sensors_dict)

    state, _ = renv.reset(StateFunction=agent.get_state)

    with tqdm(total=renv.get_timeout()) as pbar:  # we keep a progress bar
        while renv.get_info()['status'] == 'Running':

            controls = agent.step(state)
            state, _ = renv.step([controls], [state])
            pbar.update(0.05)

    # There is the possibility of saving the trajectories that the vehicles went over.
    if save_trajectories:
        renv.draw_trajectory('_trajectories')

    if draw_relative_angle_error:
        print('Saving relative angle error')
        renv.draw_relative_angle_error('_trajectories')

    info = renv.get_info()
    renv.stop()
    agent.reset()

    return info



def benchmark(benchmark_name, docker_image, gpu, agent_module, agent_params_path,
              batch_size=1, save_sensors=False, save_dataset=False, port=None,
              agent_checkpoint_name=None, non_rendering_mode=False,
              save_trajectories=False, make_videos=False,  draw_relative_angle_error=False):

    """
    Compute the benchmark for a given json file containing a certain number of experiences.

    :param benchmark_name: the name of the json file used for the benchmark
    :param docker_image: the docker image that is going to be created to perform the bench
    :param gpu: the gpu number to be used
    :param agent_module: the module of the agent class to be benchmarked.
    :param agent_class_path: the pointer to the agent that is going to be benchmarked
    :param agent_params_path: the pointer to the params file of the agent
    :param batch_size: number of repetions ( Simultaneous ) NOT IMPLEMENTED
    :param number_repetions: number of repetitions necessary
    :param save_dataset: if you are saving the data when benchmarking
    :param port: the port, in this case expect the docker to not be initialized
    :return:
    """

    params = {'save_dataset': save_dataset,
              'save_sensors': save_sensors,
              'make_videos': make_videos,
              'docker_name': docker_image,
              'gpu': gpu,
              'batch_size': batch_size,
              'remove_wrong_data': False,
              'non_rendering_mode': non_rendering_mode,
              'carla_recording': True,
              }
    env_batch = None
    # this could be joined
    while True:
        try:
            print (" STARING BENCHMARK ")
            # We reattempt in case of failure of the benchmark
            dbatch = DrivingBatch(benchmark_name, params, port=port)
            # to load CARLA and the scenarios are made
            # Here some docker was set
            dbatch.start(agent_name=agent_checkpoint_name)
            # take the path to the class and instantiate an agent
            agent = getattr(agent_module, agent_module.__name__)(agent_params_path)
            with tqdm(total=len(dbatch)) as pbar:  # we keep a progress bar
                for renv in dbatch:
                    try:
                        # Just execute the environment. For this case the rewards doesnt matter.
                        env_exec_info = benchmark_env_loop(renv, agent,
                                                           save_trajectories=save_trajectories,
                                                           draw_relative_angle_error= draw_relative_angle_error)
                        logging.debug("Finished episode got summary ")
                        # Add partial summary to allow continuation
                        add_summary(renv._environment_name, env_exec_info['summary'],benchmark_name, agent_checkpoint_name)
                        pbar.update(1)

                    except KeyboardInterrupt:
                        break
                    except Exception as e:
                        traceback.print_exc()
                        # By any exception you have to delete the environment generated data
                        renv.remove_data()
                        raise e

            del env_batch
            dbatch.cleanup()
            break
        except KeyboardInterrupt:
            del env_batch
            break
        except:
            traceback.print_exc()
            del env_batch
            break

def execute_benchmark(benchmark_file_name, docker,
                      gpu, agent_module, config, port, agent_name, non_rendering_mode,
                      save_trajectories, make_videos):
    """
        Basically execute the benchmark and save it, with respect to the parameters sent.
        This function is more about saving the benchmark than the execution itself.
    :return:
    """
    #condition, task, town_name, docker
    benchmark_file = os.path.join('version09x/descriptions', benchmark_file_name)
    print (" STARTING BENCHMARK ", benchmark_file)
    benchmark(benchmark_file, docker, gpu, agent_module, config, port=port,
                  agent_checkpoint_name=agent_name, save_sensors=True, save_dataset=True,
              non_rendering_mode=non_rendering_mode,
              save_trajectories=save_trajectories, make_videos=make_videos,  draw_relative_angle_error=False)
    # Create the results folder here if it does not exists
    if not os.path.exists('_results'):
        os.mkdir('_results')


    file_base_out = os.path.join('_results', benchmark_file_name.split('/')[-1])

    if check_benchmark_finished(benchmark_file,  agent_name):
        summary_data = check_benchmarked_episodes_metric(benchmark_file, agent_name)
        write_summary_csv(file_base_out,  agent_name, summary_data)



def benchmark_cleanup(package_name, agent_checkpoint_name):

    shutil.rmtree(os.environ["SRL_DATASET_PATH"], package_name,
                  agent_checkpoint_name)
