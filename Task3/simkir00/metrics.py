def multiclass_accuracy(prediction, ground_truth):
    
    correctly_predicted = 0
    for i in range(ground_truth.shape[0]):
        if prediction[i] == ground_truth[i]:
            correctly_predicted += 1
    accuracy = correctly_predicted / prediction.shape[0]

    return accuracy
