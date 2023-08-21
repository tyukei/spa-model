import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
# Assuming you have the data loaded as 'data'
def test_lstm_model(test_path):
    graph(test_path)
    raw_test_data = pd.read_csv(test_path)
    time_info = raw_test_data['時刻'].iloc[seq_length:].values

    X_test, y_test = preprocess_data(test_path)
    y_pred = model.predict(X_test)
    y_pred_labels_mapped = np.argmax(y_pred, axis=1)
    y_pred_labels = np.array([reverse_label_mapping[label] for label in y_pred_labels_mapped])
    plt.figure(figsize=(15, 6))

    # Primary plot for Actual vs Predicted Labels
    ax1 = plt.gca() # Gets the current Axes instance
    ax1.scatter(time_info, y_test, label='Actual Label', marker='o', linestyle='-')
    ax1.scatter(time_info, y_pred_labels, label='Predicted Label', marker='.', linestyle='--')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Label')
    ax1.legend(loc='upper left')
    ax1.tick_params(axis='y', colors='black') # Ensure y-axis ticks match color of primary plot
    ax1.set_xticks(ax1.get_xticks()[::10]) # Select every 10th tick for x-axis
    plt.xticks(rotation=45)

    surface_temp = raw_test_data['体表温度'].iloc[seq_length:].values
    ax2 = ax1.twinx()  # Create a twin axis
    ax2.plot(time_info, surface_temp, label='体表温度', color='green')
    ax2.set_ylabel('体表温度', color='green')
    ax2.tick_params(axis='y', colors='green') # Ensure y-axis ticks match color of 体表温度 plot


    xticks_space = 100
    plt.xticks(np.arange(0, len(time_info), xticks_space))
    plt.xlim([0, len(time_info)])

    plt.title('Actual vs Predicted Labels over Time and 体表温度')
    plt.tight_layout()
    plt.show()
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred_labels)

    # Define class names for your data
    classes = [reverse_label_mapping[i] for i in range(len(reverse_label_mapping))]

    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 7))
    plot_confusion_matrix(cm, classes=classes, normalize=True, title='Normalized confusion matrix')
    plt.show()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        normalized_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        normalized_cm = cm

    print(normalized_cm)

    plt.imshow(normalized_cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = normalized_cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{} ({:.2f}%)".format(cm[i, j], normalized_cm[i, j] * 100),
                 horizontalalignment="center",
                 color="white" if normalized_cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

