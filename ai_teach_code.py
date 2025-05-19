def launch_fe(data):
    import os
    import pandas as pd
    import numpy as np
    from io import StringIO
    import json
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.feature_extraction import text
    import pickle
    from scipy import sparse
    MAX_TEXT_FEATURES = 200
    columns_list = ["num_countries", "who_am_I", "height", "years_school"]

    dataset = pd.read_csv(data, skipinitialspace=True)

    # Replace inf and -inf, with max and min values of the particular column
    df = dataset.select_dtypes(include=np.number)
    cols = df.columns.to_series()[np.isinf(df).any()]
    col_min_max = {np.inf: dataset[cols][np.isfinite(dataset[cols])].max(), -np.inf: dataset[cols][np.isfinite(dataset[cols])].min()} 
    dataset[cols] = dataset[cols].replace({col: col_min_max for col in cols})

    num_samples = len(dataset)

    # Encode labels into numbers starting with 0
    label = "who_am_I"
    tmpCol = dataset[label].astype('category')
    dict_encoding = { label: dict(enumerate(tmpCol.cat.categories.astype(str)))}
    print('dict_encoding', dict_encoding)
    # Save the model
    model_name = "60bbfe26-4437-41d7-9985-57a85fa22388"
    fh = open(model_name, "wb")
    pickle.dump(dict_encoding, fh)
    fh.close()

    label = "who_am_I"
    dataset[label] = tmpCol.cat.codes

    # Move the label column
    cols = list(dataset.columns)
    colIdx = dataset.columns.get_loc("who_am_I")
    # Do nothing if the label is in the 0th position
    # Otherwise, change the order of columns to move label to 0th position
    if colIdx != 0:
        cols = cols[colIdx:colIdx+1] + cols[0:colIdx] + cols[colIdx+1:]
        dataset = dataset[cols]

    # split dataset into train and test
    train, test = train_test_split(dataset, test_size=0.2, random_state=42)

    # Write train and test csv
    train.to_csv('train.csv', index=False, header=False)
    test.to_csv('test.csv', index=False, header=False)
    column_names = list(train.columns)
def get_model_id():
    return "60bbfe26-4437-41d7-9985-57a85fa22388"

# Please replace the brackets below with the location of your data file
data = '<>'

launch_fe(data)

# import the library of the algorithm
from sklearn.neural_network import MLPClassifier

# Initialize hyperparams
hidden_layer_sizes = (100,)
activation = 'relu'
solver = 'adam'
alpha = 0.0001
learning_rate = 'constant'
learning_rate_init = 0.001
max_iter = 200
early_stopping = True

# Initialize the algorithm
model = MLPClassifier(random_state=1, hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha, learning_rate=learning_rate, learning_rate_init=learning_rate_init, max_iter=max_iter, early_stopping=early_stopping)
algorithm = 'MLPClassifier'

import pandas as pd
# Load the test and train datasets
train = pd.read_csv('train.csv', skipinitialspace=True, header=None)
test = pd.read_csv('test.csv', skipinitialspace=True, header=None)
# Train the algorithm
model.fit(train.iloc[:,1:], train.iloc[:,0])
def encode_confusion_matrix(confusion_matrix):
    import pickle
    encoded_matrix = dict()
    object_name = get_model_id()
    file_name = open(object_name, 'rb')
    dict_encoding = pickle.load(file_name)
    labels = list(dict_encoding.values())[0]
    for row_indx, row in enumerate(confusion_matrix):
        encoded_matrix[labels[row_indx]] = {}
        for item_indx, item in enumerate(row):
            encoded_matrix[labels[row_indx]][labels[item_indx]] = item
    return encoded_matrix

# Predict the class labels
y_pred = model.predict(test.iloc[:,1:])
# import the libraries to calculate confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# calculate confusion matrix
encode_label = {"who_am_I": {"0": "adult", "1": "child"}}
predicted_label = [*encode_label][0]
keys_list = encode_label[predicted_label].keys()
labels_display = []
labels = [int(x) for x in keys_list]
for a in range(0,len(keys_list)):
    labels_display.append(encode_label[predicted_label][str(a)])
calculated_confusion_matrix = confusion_matrix(test.iloc[:,0], y_pred, labels=labels)
cmd = ConfusionMatrixDisplay(calculated_confusion_matrix, display_labels = labels_display)
cmd.plot(cmap=plt.cm.Blues)
# calculate accuracy
score = model.score(test.iloc[:, 1:], test.iloc[:, 0])
# The value is returned as a decimal value between 0 and 1
# converting to percentage
accuracy = score * 100
print('Accuracy of the model is: ', accuracy)

# fe_transform function traansforms raw data into a form the model can consume
print('Below is the prediction stage of the AI')
def fe_transform(data_dict, object_path=None):
    import os
    import pandas as pd
    from io import StringIO
    import json
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.feature_extraction import text
    import pickle
    from scipy import sparse
    
    dataset = pd.DataFrame([data_dict])

    return dataset
def encode_label_transform_predict(prediction):
    import pickle
    encoded_prediction = prediction
    label = "who_am_I"
    object_name = "60bbfe26-4437-41d7-9985-57a85fa22388"
    file_name = open(object_name, 'rb')
    dict_encoding = pickle.load(file_name)
    label_name = list(dict_encoding.keys())[0]
    encoded_prediction = \
        dict_encoding[label_name][int(prediction)]
    print(encoded_prediction)
    return encoded_prediction
def get_labels(object_path=None):
    label_names = []
    label_name = list(dict_encoding.keys())[0]
    label_values_dict = dict_encoding[label_name]
    for key, value in label_values_dict.items():
        label_names.append(str(value))

test_sample = {'height': 4.175, 'years_school': 9.5, 'num_countries': 5}
# Call FE on test_sample
test_sample_modified = fe_transform(test_sample)
# Make a prediction
prediction = model.predict(test_sample_modified)
encode_label_transform_predict(prediction)

# Calculate and plot roc-auc
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

encode_label = {"who_am_I": {"0": "adult", "1": "child"}}
categories = np.unique(test.iloc[:, 0])

if algorithm and algorithm == 'LinearSVC':
    predictions = model.decision_function(test.iloc[:, 1:])
else:
    predictions = model.predict_proba(test.iloc[:, 1:])
    predictions = predictions[:, 1] 

if encode_label:
    fpr, tpr, _ = roc_curve(test.iloc[:, 0], predictions)
else:
    pos_label = categories[1]
    fpr, tpr, _ = roc_curve(test.iloc[:, 0], predictions, pos_label)

auc = roc_auc_score(test.iloc[:, 0], predictions)
#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(bbox_to_anchor=(1, 1.12))
plt.show()
