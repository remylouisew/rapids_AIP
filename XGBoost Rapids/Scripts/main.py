import argparse
import logging

from train import train


def main(args):
    logging.info("starting the train job")
    model = train(args)

    logging.info("Finish distributed XGBoost job")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train-files',
        type=str,
        help='Training files local or GCS',
        default='/tmp/higgs/*.csv')
    parser.add_argument(
        '--scheduler-ip-file',
        type=str,
        help='scratch temp file to storage scheduler ip in GCS',
        default='gs://dlvm-dataset/tmp/scheduler.txt')
    parser.add_argument(
        '--n_estimators',
        help='Number of trees in the model',
        type=int,
        default=100
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        help='num of workers for rabit')

    logging.basicConfig(format='%(message)s')
    logging.getLogger().setLevel(logging.INFO)
    main_args = parser.parse_args()
    main(main_args)
