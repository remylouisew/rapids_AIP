import logging, time
import os, gc
import multiprocessing as mp
import cudf
import json
import xgboost as xgb
import socket

logger = logging.getLogger(__name__)

def extract_xgbooost_cluster_env(scheduler_ip_file):
    """
    Extract the cluster env from pod
    :return: the related cluster env to build rabit
    """
    logger.info("starting to extract system env")
    tf_config_str = os.environ.get('TF_CONFIG')
    tf_config_json = json.loads(tf_config_str)
    print(tf_config_json)
    task_name = tf_config_json.get('task', {}).get('type')
    
    if task_name == 'worker': 
        rank = int(tf_config_json.get('task', {}).get('index')) + 1
        while not os.path.exists(scheduler_ip_file):
            time.sleep(1)
        scheduler_ip = ''
        while scheduler_ip == '':
            with open(scheduler_ip_file, 'r') as f:
                scheduler_ip = f.read().rstrip("\n")    
    else:
        rank = 0
        host_name = socket.gethostname()
        scheduler_ip = socket.gethostbyname(host_name)        
        with open(scheduler_ip_file, 'w') as f:
            f.write(scheduler_ip)
    
    master_addr = scheduler_ip 
    master_port = 9091

    logger.info("extract the Rabit env from cluster :"
                " %s, port: %d, rank: %d",
                master_addr, master_port, rank)

    return master_addr, master_port, rank


def _cudf_read_csv(gcs_file):
    colnames = ['label'] + ['feature-%02d' % i for i in range(1, 29)]
    """
    :param gcs_file: gcs file location
    :return: cudf dataframe
    """
    return cudf.io.csv.read_csv(gcs_file, header=None, names=colnames)


def gpu_process(assigned_files, ctx, nthreads):
    """
    :param assigned_files: assigned csv files on work
    :param ctx: process handle
    :param nthreads: number of threads
    :return:
    """
    pool = ctx.Pool(nthreads)
    data_frames = pool.map(
        _cudf_read_csv,
        assigned_files)
    pool.close()
    pool.join()
    return data_frames


def read_train_data(train_dir, rank):
    """
    Read file based on the rank of worker. We have 480 training files, each sized between 1.2GB-1.6GB. We have each
    worker read 5 files (in fewer node setting, seems like each work can read 6 files, that changes when we scale it
    out). To do the experiment on a larger scale, we reuse training files by having multiple worker read from it.
    GPU adding file reader thread won't help given the limited CPU resources
    :param rank: the id of each worker
    :param assigned_files: the input file name or the place to read the data
    :return: XGBoost Dmatrix
    """
    try:
        ctx = mp.get_context("spawn")
    except RuntimeError:
        pass
    logging.info("Read data to DMatrix")
    nthreads = 2 
    assigned_files = [train_dir + "{:04d}.csv".format(i) for i in range(rank*12,(rank+1)*12)]
    gpu_dfs = gpu_process(assigned_files, ctx, nthreads)

    df = cudf.concat(gpu_dfs, ignore_index=True)
    dtrain = xgb.DeviceQuantileDMatrix(data=df[df.columns.difference(['label'])], label=df[['label']])
    return dtrain
