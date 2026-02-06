# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Load the dataset and separate features X and label y 
2.Divide the dataset into training and testing sets 
3.Fit a multinomial logistic regression model on the training data. 
4.Predict on the test set and calculate accuracy (optionally, display the confusion matrix). 
```
## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Anikha Pillai
RegisterNumber: 25009524 
*/
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
data = load_iris()
x, y = data.data, data.target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

```

## Output:
<img width="197" height="56" alt="image" src="https://github.com/user-attachments/assets/c37b4cc2-bb56-43e0-b305-435b865f5937" />
<img width="701" height="580" alt="image" src="https://github.com/user-attachments/assets/ceb81d1c-2417-418c-9b84-9f3b9def7b55" />
<img width="701" height="580" alt="Screenshot 2026-02-06 113506" src="https://github.com/user-attachments/assets/ac58e878-fc4f-4a9a-bdc5-304722b8f706" />





## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
