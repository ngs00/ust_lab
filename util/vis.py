import numpy
import matplotlib.pyplot as plt


def plot_pred_results(tgts, preds):
    targets_test = numpy.concatenate(tgts)
    preds_test = numpy.concatenate(preds)
    plt.grid(linestyle='--')
    plt.scatter(targets_test, preds_test)
    plt.xlabel('Target value')
    plt.ylabel('Predicted value')
    plt.show()
    plt.close()
