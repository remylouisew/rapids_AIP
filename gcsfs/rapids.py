import time, argparse
import subprocess, sys, os, json
import dask, dask_cudf, asyncio 
import socket, gcsfs
from dask.distributed import Client
import xgboost as xgb

async def start_client(scheduler_addr, train_dir, model_file, num_workers, fs):
  async with Client(scheduler_addr, asynchronous=True) as client:
    dask.config.set({'distributed.scheduler.work-stealing': False})
    print(dask.config.get('distributed.scheduler.work-stealing'))
    dask.config.set({'distributed.scheduler.bandwidth': 1})
    print(dask.config.get('distributed.scheduler.bandwidth'))
    await client.wait_for_workers(num_workers)
    colnames = ['label'] + ['feature-%02d' % i for i in range(1, 29)]
    df = dask_cudf.read_csv(train_dir, header=None, names=colnames, chunksize=None)
    start_time = time.time()
    dtrain = await xgb.dask.DaskDeviceQuantileDMatrix(client, df[df.columns.difference(['label'])], df['label'])
    output = await xgb.dask.train(client,
                        { 'verbosity': 2,
                         'learning_rate': 0.1,
                          'max_depth': 8,
                          'objective': 'reg:squarederror',
                          'subsample': 0.6,
                          'gamma': 1,
                          'verbose_eval': True,
                          'tree_method':'gpu_hist',
                          'nthread': 1
                        },
                        dtrain,
                        num_boost_round=100, evals=[(dtrain, 'train')])
    print("[debug:leader]: ------ training finished")
    output['booster'].save_model('/tmp/tmp.model')
    history = output['history']
    print('[debug:leader]: ------ Training evaluation history:', history)
    fs.put('/tmp/tmp.model', model_file)
    print("[debug:leader]: ------model saved")
    print("[debug:leader]: ------ %s seconds ---" % (time.time() - start_time))
    output = await xgb.dask.train(client,
                        { 'verbosity': 2,
                         'learning_rate': 0.1,
                          'max_depth': 8,
                          'objective': 'reg:squarederror',
                          'subsample': 0.5,
                          'gamma': 0.9,
                          'verbose_eval': True,
                          'tree_method':'gpu_hist',
                          'nthread': 1
                        },
                        dtrain,
                        num_boost_round=100, evals=[(dtrain, 'train')])
    print("[debug:leader]: ------ training finished")
    output['booster'].save_model('/tmp/tmp.model')
    history = output['history']
    print('[debug:leader]: ------ Training evaluation history:', history)
    fs.put('/tmp/tmp.model', model_file + '2')
    print("[debug:leader]: ------model saved")
    print("[debug:leader]: ------ %s [2nd]seconds ---" % (time.time() - start_time))
    await client.shutdown()

def launch_dask(cmd, is_shell):
  return subprocess.Popen(cmd,
                    stdout=None,
                    stderr=None,
                    shell=is_shell)

def launch_worker(cmd):
  return subprocess.check_call(cmd,
                    stdout=sys.stdout,
                    stderr=sys.stderr)

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--gcp-project',
    type=str,
    help='user gcp project',
    default='remy-demos')
  parser.add_argument(
    '--train-files',
    type=str,
    help='Training files local or GCS',
    default='gs://nvidiadask/higgs/*.csv')
  parser.add_argument(
    '--scheduler-ip-file',
    type=str,
    help='scratch temp file to storage scheduler ip in GCS',
    default='gs://nvidiadask/tmp/scheduler.txt')
  parser.add_argument(
    '--model-file',
    type=str,
    help="""GCS or local dir for checkpoints, exports, and summaries.
    Use an existing directory to load a trained model, or a new directory
    to retrain""",
    default='gs://nvidiadask/models/001.model')
  parser.add_argument(
    '--num-workers',
    type=int,
    help='num of workers for rabit')
  args, _ = parser.parse_known_args()
  tf_config_str = os.environ.get('TF_CONFIG')
  tf_config_json = json.loads(tf_config_str)
  print(tf_config_json)
  task_name = tf_config_json.get('task', {}).get('type')
  # scheduler_ip = tf_config_json.get('cluster', {}).get('master')[0]
  fs = gcsfs.GCSFileSystem(project=args.gcp_project, token='cloud')
  if task_name == 'master':
    host_name = socket.gethostname()
    host_ip = socket.gethostbyname(host_name) 
    with fs.open(args.scheduler_ip_file, 'w') as f:
      f.write(host_ip)
    scheduler_addr = host_ip + ':8786'
    proc_scheduler = launch_dask(f'dask-scheduler --protocol tcp > /tmp/scheduler.log 2>&1 &', True)
    print('[debug:leader]: ------ start scheduler')
    proc_worker = launch_dask(['dask-cuda-worker', '--rmm-pool-size', '8G', '--nthreads', '2', scheduler_addr], False) 
    print('[debug:leader]: ------ start worker')
    asyncio.get_event_loop().run_until_complete(start_client(scheduler_addr, 
                             args.train_files, 
                             args.model_file,
                             args.num_workers, fs))
    fs.rm(args.scheduler_ip_file, recursive=True)
  # launch dask worker, redirect output to sys stdout/err
  elif task_name == 'worker':
    while not fs.exists(args.scheduler_ip_file):
      time.sleep(1)
    with fs.open(args.scheduler_ip_file, 'r') as f:
      scheduler_ip = f.read().rstrip("\n")
    print('[debug:scheduler_ip]: ------'+scheduler_ip)
    scheduler_addr = scheduler_ip + ':8786'
    proc_worker = launch_worker(['dask-cuda-worker', '--rmm-pool-size', '8G', '--nthreads', '2', scheduler_addr])
