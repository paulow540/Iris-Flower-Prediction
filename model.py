import seaborn as sns
import joblib as jb 
iris = sns.load_dataset('iris')
iris.tail()
X_iris = iris.drop('species', axis=1)

y_iris = iris['species']



from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris, random_state=1, train_size=0.8)



from sklearn.naive_bayes import GaussianNB     # 1. choose model class
model = GaussianNB() # 2. instantiate model
model.fit(Xtrain, ytrain) # 3. fit model to data
y_model = model.predict(Xtest) # 4. predict on new data


# Finally, we can use the accuracy_score utility to see the fraction of predicted labels
# that match their true value:
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest, y_model))

# Save the model to a file
jb.dump(model, 'iris_model.pkl')