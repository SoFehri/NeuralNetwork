import numpy as np
import argparse
from NeuralNetwork import *
#from utils import mnist_reader


parser = argparse.ArgumentParser()
parser.add_argument('-test_percent', required = True, type = float, help="Test percentage")
parser.add_argument('-valid_percent', type = float, help="Validation Percentage")
parser.add_argument('-image', type=bool, help = "Specify the type as image")
parser.add_argument('-mnist', type=bool, help = "Specify that we're treating mnist images")
parser.add_argument('-csv', type=bool, help = "Specify the type as csv")
parser.add_argument('-path', type=str, required = True, help="Path of data")
parser.add_argument('-sizes', type=int, required = True, nargs='+', help="Number of layers and number of nurones per layer as list")
parser.add_argument('-epochs', type=int, required = True, help="Number of epochs")
parser.add_argument('-lrate', type=float, required = True, help="learning rate")
parser.add_argument('-minibatch_size', type=int, help="minibatch_size default = 100 || if 0 then we use all the dataset (Batch)")
parser.add_argument('-L1_lambda', type=float, help="Lambda for L1 regularization default = 0 \
                      L1 and L2 gives elastic net regularization")
parser.add_argument('-L2_lambda', type=float, help="Lambda for L2 regularization default = 0")
parser.add_argument('-h_act', type=str, help="Hidden layer activation default relu")
parser.add_argument('-out_act', type=str, help="Output layer activation default softmax")
parser.add_argument('-test', type=str, help="Test default False")

args = parser.parse_args()

def main():
    #load data
    valid_percent = 0
    image = False
    mnist = True
    csv = False
    lambda1 = 0
    lambda2 = 0
    h_act = 'relu'
    out_act = 'softmax'
    test = False
    minibatch_size = 100
    images = True

    path = args.path
    sizes = args.sizes
    epochs = args.epochs
    lrate  = args.lrate
    test_percent = args.test_percent/100

    if args.valid_percent : valid_percent = args.valid_percent/100
    if args.image         : image = args.image
    if args.mnist         : mnist = args.mnist
    if args.csv           : csv = args.csv
    if args.L1_lambda     : lambda1 = args.L1_lambda
    if args.L2_lambda     : lambda2 = args.L2_lambda
    if args.minibatch_size: minibatch_size = args.minibatch_size
    if args.h_act     : h_act = args.h_act
    if args.out_act     : out_act = args.out_act
    if args.test     : test = args.test

    if csv or 'txt' in path or '.csv' in path :
        images = False
        data = load_csv(path,test_percent,valid_percent)

    elif mnist == False:
        data = load_images(path,test_percent,valid_percent,mnist = False) 
    else :
        data = load_images(path,test_percent,valid_percent) 

    model = NeuralNetwork(sizes)
    model.SGD(data, epochs , minibatch_size, lambda1, lambda2, lrate, h_act, out_act , test, images)
    

def load_images(path,test_percent,valid_percent,mnist=True):
    datasets = {}
    if mnist:
        x_train, y_train = mnist_reader.load_mnist(path, kind='train')
        x_test, y_test = mnist_reader.load_mnist(path, kind='t10k')
        train_labels = y_train.reshape((y_train.shape[0],1))
        test_labels = y_test.reshape((y_test.shape[0],1))

        train_inputs = x_train/255
        test_inputs  = x_test/255

        datasets = {'train_inputs':train_inputs,'train_labels':train_labels,
                'test_inputs':test_inputs,'test_labels':test_labels}

        if valid_percent > 0 :
            train_inputs, validation_inputs = create_datasets(valid_percent,train_inputs)
            train_labels, validation_labels = create_datasets(valid_percent,train_labels)

            datasets = {'train_inputs':train_inputs,'train_labels':train_labels,'validation_inputs':validation_inputs,
                    'validation_labels':validation_labels,'test_inputs':test_inputs,'test_labels':test_labels}

    return datasets


def load_csv(path,test_percent,valid_percent):
    data = np.loadtxt(path)
    train_inputs = data[:,:-1]
    train_labels = data[:,-1]
    train_inputs,test_inputs = create_datasets(train_inputs,test_percent)
    train_labels,test_labels = create_datasets(train_labels,test_percent)
    datasets = {'train_inputs':train_inputs,'train_labels':train_labels,
                'test_inputs':test_inputs,'test_labels':test_labels}
    if valid_percent > 0:
        train_inputs, validation_inputs = create_datasets(train_inputs,valid_percent)
        train_labels,validation_labels = create_datasets(train_labels,valid_percent)
        datasets = {'train_inputs':train_inputs,'train_labels':train_labels,'validation_inputs':validation_inputs,
                    'validation_labels':validation_labels,'test_inputs':test_inputs,'test_labels':test_labels}

    return datasets

def create_datasets(data, percent):
    idx = int(percent * len(data))
    return data[:idx], data[idx:]


if __name__ == '__main__':
    main()