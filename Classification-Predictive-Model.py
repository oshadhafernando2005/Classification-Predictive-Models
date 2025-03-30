import pandas as pd
# load dataset
pima = pd.read_csv('/content/diabetes.csv')
pima.head()
#split dataset in features and target variable
feature_cols = ['Pregnancies', 'Insulin',
'Age','Glucose','BMI','DiabetesPedigreeFunction']
X = pima[feature_cols] # Features
y = pima.Outcome # Target variable
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X1 = scaler.fit_transform(X)
# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X1_train,X1_test,y_train,y_test=train_test_split(X1,y,test_size=0.25,random_state=0)
print('Whole Data shape', pima.shape)
print('X1_train shape', X1_train.shape)
print('X1_test shape', X1_test.shape)
# import the class
from sklearn.linear_model import LogisticRegression
# instantiate the model (using the default parameters)
logreg = LogisticRegression()
logreg.fit(X1_train, y_train)
y_pred=logreg.predict(X1_test)
Comparison_df = pd.DataFrame({'Actual Diabetic Diagnoses' : y_test,
'Predicted' : y_pred})
Comparison_df.to_csv(r'/content/Diagnoses_Comparison_df.csv', index=True)
Comparison_df
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 200)
# To plot the confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
# Construct the confusion matrix cm
cm = confusion_matrix(y_test, y_pred, labels=logreg.classes_)
# Create a display to plot the confusion matrix
disp = ConfusionMatrixDisplay(cm,display_labels=logreg.classes_)
disp.plot()



# Import the function to calculate accuracy score
from sklearn.metrics import accuracy_score
# Apply the function to find the correct predictions
accuracy = accuracy_score(y_test,y_pred)
# Display the accuracy
print ('The Logistic Regression Model Accuracy:',accuracy)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Import the function from the package
from sklearn.metrics import RocCurveDisplay
# Apply the function by specifying the name of your model and test data.
Logreg_roc = RocCurveDisplay.from_estimator(logreg, X1_test, y_test)

























