import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier

# Prepering the data
df = pd.read_csv(r'E:\Nour\Machine Learning\projects\Fault Detection\detect_dataset.csv')
values  = df.iloc[:, 1:7]
outputs = df['Output (S)']
values_array  = values.to_numpy() 
outputs_array = outputs.to_numpy()
print(f"Dataset size : {outputs_array.shape[0]} training example")

# Creating and training the model
detector = MLPClassifier(hidden_layer_sizes= (25, 15, 10), solver = 'adam', max_iter = 200, tol = 1e-5)
vtrain, vtest, otrain, otest = train_test_split(values_array, outputs_array, test_size = .2)
fitted_model = detector.fit(vtrain, otrain) # Training the model on the data

# Making prediction 
predictions = fitted_model.predict(vtest) 

# Error analysis
label = ['No Fault', 'Fault']
conf_mat  = confusion_matrix(otest, predictions)
report = classification_report(otest, predictions, target_names = label)
print(f"Confusion Matrix is like \n {conf_mat}")
print("===================================")
print(report)

# Drawing the confusion matrix
confusion_mat_graph = plt.matshow(conf_mat)
plt.show()