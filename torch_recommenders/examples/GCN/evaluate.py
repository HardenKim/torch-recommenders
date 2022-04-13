
def accuracy(output, labels):
    correct_preds = output.argmax(1).type_as(labels) == labels
    acc = correct_preds.float().mean()

    return acc