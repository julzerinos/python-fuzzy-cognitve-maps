import fcm_test
import fcm_train
import argparse


def main(args):
    print(args)
    ts = fcm_train.main(args)
    fcm_test.main(ts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fuzzy Cognitive Mapping training and testing')
    parser.add_argument('-s', '--step', dest='step', default="overlap", choices=["overlap", "distinct"], type=str, help='Steps for training')
    parser.add_argument('-t', '--transformation', dest='transform', default="sigmoid",
     choices=["sigmoid", "binary", "tanh", "arctan", "gaussian"],
     type=str, help='Transformation function')
    parser.add_argument('-e', '--error', dest='error', default="rmse",
     choices=["rmse", "mpe", "max_pe"], type=str, help='Error function')  
    parser.add_argument('-m', '--mode', dest='mode', default="outer",
     choices=["outer", "inner"], type=str, help='Mode of calculations')
    parser.add_argument("-i", "--iter", dest="iter", default=500, type=int, help='Training iterations')
    parser.add_argument("-p", "--performance", dest="pi", default=1e-5, type=float, help='Performance index')

    args = parser.parse_args()
    argu = args.step, args.transform, args.error, args.mode, args.iter, args.pi
    main(argu)
