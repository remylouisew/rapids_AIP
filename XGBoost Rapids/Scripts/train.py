import logging, time
import xgboost as xgb
import traceback
import rmm
import os
from tracker import RabitTracker
from utils import read_train_data, extract_xgbooost_cluster_env

logger = logging.getLogger(__name__)

def train(args):
    """
    :param args: configuration for train job
    :return: XGBoost model
    """
    # get env from xgboost operator
    start = time.time()
    addr, port, rank = extract_xgbooost_cluster_env(args.scheduler_ip_file)
    world_size = args.num_workers
    n_estimators = args.n_estimators
    rabit_tracker = None

    initial_pool_size = 7 << 30

    rmm.reinitialize(
        pool_allocator=True,
        managed_memory=True,
        initial_pool_size=initial_pool_size,
        logging=False,
    )

    try:
        """start to build the network"""
        if world_size > 1:
            if rank == 0:
                logger.info("start the master node")

                rabit = RabitTracker(hostIP="0.0.0.0", nslave=world_size)
                rabit.start(world_size)
                rabit_tracker = rabit
                logger.info('###### RabitTracker Setup Finished ######')

            envs = [
                'DMLC_NUM_WORKER=%d' % world_size,
                'DMLC_TRACKER_URI=%s' % addr,
                'DMLC_TRACKER_PORT=%d' % port,
                'DMLC_TASK_ID=%d' % rank,
                'DMLC_WORKER_CONNECT_RETRY=99999999' 
            ]
            logger.info('##### Rabit rank setup with below envs #####')
            for i, env in enumerate(envs):
                logger.info(env)
                envs[i] = str.encode(env)

            xgb.rabit.init(envs)
            logger.info('##### Rabit rank = %d' % xgb.rabit.get_rank())
            rank = xgb.rabit.get_rank()

        else:
            world_size = 1
            logging.info("Start the train in a single node")

        logger.info('Init with DMLC rabit: ' + str(time.time() - start))
        start = time.time()
        dmatrix = read_train_data(args.train_files, rank=rank)
        logger.info('IO time with cudf: ' + str(time.time() - start))
        start = time.time()

        kwargs = {}
        kwargs["dtrain"] = dmatrix
        kwargs["num_boost_round"] = int(n_estimators)
        param_xgboost_default = {'learning_rate':0.3, 'max_depth': 8, 'silent': True,
                                 'objective': 'reg:squarederror', 'subsample': 0.1, 'gamma': 1,
                                 'verbose_eval': True, 'tree_method':'gpu_hist'}

        kwargs["params"] = param_xgboost_default

        logging.info("starting to train xgboost at node with rank %d", rank)
        bst = xgb.train(**kwargs)
        logger.info('xgboost training time with cudf and rabit on GPU: ' + str(time.time() - start))

        if rank == 0:
            model = bst
        else:
            model = None

        logging.info("finish xgboost training at node with rank %d", rank)

    except Exception as e:
        logger.error("something wrong happen: %s", traceback.format_exc())
        raise e
    finally:
        logger.info("xgboost training job finished!")
        if world_size > 1:
            xgb.rabit.finalize()
        if rabit_tracker:
            rabit_tracker.join()

    return model
