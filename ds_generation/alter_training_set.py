import argparse
import numpy as np


def perturb_labels(labs, p = 0.05):

    perturbed_labs = []
    for lab in labs:
        perturb = np.random.binomial(1, p)
        if perturb == 1:
            perturbed_labs.append(1 - lab)
        else:
            perturbed_labs.append(lab)
    return np.array(perturbed_labs)


def main(args):
    
    
    
    X = np.load(args.features)
    y = np.load(args.labels, allow_pickle = True)

    X_train, X_test, y_train, y_test = tts(X, y, random_state = 0)
    
    
    perturb_prob = [1, 2, 5, 10, 20, 30, 40, 50]
    training_size = [100, 1000, 5000, 10000, 20000, 50000, 100000, 150000]
    
    
    for p in perturb_prob:
        for n in training_size:
            X_subset = X_train[:n, :]
            y_subset = y_train[:n]
            
            y_
    
    



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('features', type=str, help='Numpy array of features')
    parser.add_argument('labels', type = str, help = 'Numpy array of labels')
    parser.add_argument('output_dir', type=str,
                        help='Location to save output')
    
    
    

    arguments = parser.parse_args()
    
    main(arguments)