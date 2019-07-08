import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
file = "world-happiness-report-2019.csv"
columns = ["Country", "Ladder", "Positive affect", "Negative affect", "Social support", "Freedom", "Corruption", "Genorosity", "Healthy life expectancy"]
dataset = pd.read_csv(file, names=columns)
data = dataset.values
# X contains all information except for countries and ladder
X = data[1:, 2:]
# Y contains all countries
Y = data[1:, 0]
# Logistic regression for the model. Liblinear for one vs rest data
model = LR(solver='liblinear', multi_class='auto')
# Train the model
model.fit(X, Y)
# Try changing some parameters!
sample_test = [[56, 66, 32, 69, 21, 68, 4]]
print(model.predict(sample_test))
