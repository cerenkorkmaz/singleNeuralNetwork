import numpy as np # 1.18.0
import pandas as pd # 0.25.3
import matplotlib.pyplot as plt # 3.1.2
#Python 3.6.10


def sigmoid(x):
    return 1 / (1 + np.exp(-0.005*x))

def sigmoid_derivative(x):
    return 0.005 * sigmoid(x) * (1 - sigmoid(x) )

def read_and_divide_into_train_and_test(csv_file):

    csv_file = csv_file[~(csv_file['Bare_Nuclei'] == '?' )].astype(int)

    heatmap(csv_file)

    length=len(csv_file)
    col=len(csv_file.columns)-1
    per80=(length*80)//100

    training_inputs=np.array(csv_file.iloc[:per80,:col])
    test_inputs=np.array(csv_file.iloc[per80:,:col])

    training_labels=np.array(csv_file.iloc[:per80,col:])
    test_labels=np.array(csv_file.iloc[per80:,col:])

    return training_inputs, training_labels, test_inputs, test_labels


def run_on_test_set(test_inputs, test_labels, weights):

    tp = 0
    test_predictions=sigmoid(np.dot(test_inputs,weights))
    test_predictions=map(lambda x: 0 if x<0.5 else 1, test_predictions)

    for predicted_val, label in zip(test_predictions, test_labels):
        if predicted_val == label:
            tp += 1
    accuracy =  tp / len(test_labels)

    return accuracy


def plot_loss_accuracy(accuracy_array, loss_array):

    it=np.arange(2500)

    plt.plot(it,accuracy_array,label="Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("#Epochs")

    plt.plot(it,loss_array,label="Loss")
    plt.ylabel("Loss")
    plt.xlabel("#Epochs")
    plt.legend()

    plt.show()

def heatmap(csv_file):

    csv_file = csv_file.drop('Class',axis=1)
    corr_Matrix=csv_file.corr()

    fig=plt.figure(figsize=(5,5))
    plt.matshow(corr_Matrix,fignum=fig.number)
    plt.xticks(range(csv_file.shape[1]),csv_file.columns,fontsize=5,rotation=45)
    plt.yticks(range(csv_file.shape[1]),csv_file.columns,fontsize=5,rotation=0)
    colors=plt.colorbar()
    colors.ax.tick_params(labelsize=8)
    plt.show()


def main():

    csv_file = pd.read_csv("breast-cancer-wisconsin.csv",index_col=0)

    iteration_count = 2500
    np.random.seed(1)
    weights = 2 * np.random.random((9, 1)) - 1
    accuracy_array = []
    loss_array = []
    training_inputs, training_labels, test_inputs, test_labels = read_and_divide_into_train_and_test(csv_file)

    for iteration in range(iteration_count):
        input = training_inputs
        output=sigmoid(np.dot(input,weights))
        loss=training_labels-output
        tuning=loss*sigmoid_derivative(output)
        weights+=np.dot((training_inputs.transpose()),tuning)

        accuracy_array.append(run_on_test_set(test_inputs,test_labels,weights))
        loss_array.append(np.mean(loss))

    plot_loss_accuracy(accuracy_array, loss_array)

if __name__ == '__main__':
    main()
