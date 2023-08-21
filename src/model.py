TRAIN_FOLDER = "Train0817"
TEST_FOLDER = 'TrainAll'

import graph
import train
import test

def train_test(epoch = 1):
    train_files=train.get_files(TRAIN_FOLDER)
    for csv_file in train_files:
        print(csv_file)
        print('-------------------------------')
        train.train_lstm_model(csv_file,isGraph=True,epoch=epoch)

    test_files = test.get_files(TEST_FOLDER)
    for test_file in test_files:
        print(test_file)
        print('-------------------------------')
        test.test_lstm_model(test_file)


    print("train data")
    for csv_file in train_files:
        print(csv_file)

    print("test data")
    for test_file in test_files:
        print(test_file)

def main():
    train_test.train_test(epoch=1)


if __name__ == '__main__':
    main()