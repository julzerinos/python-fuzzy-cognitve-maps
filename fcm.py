import fcm_test
import fcm_train


def main():
    ts = fcm_train.main()
    fcm_test.main(ts)


if __name__ == '__main__':
    main()
