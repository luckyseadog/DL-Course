import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    
    '''
    
    
    TP = np.count_nonzero(prediction[prediction == ground_truth])
    TN = len(set(np.where(prediction == 0)[0]) & set(np.where(ground_truth == 0)[0]))
    FP = len(set(np.where(prediction == 1)[0]) & set(np.where(ground_truth == 0)[0]))
    FN = len(set(np.where(prediction == 0)[0]) & set(np.where(ground_truth == 1)[0]))
    
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    accuracy = (TP+TN)/(TP+TN+FN+FP)
    f1 = (2*precision*recall)/(precision+recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    return sum(prediction == ground_truth)/len(ground_truth)
