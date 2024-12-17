import pandas as pd
import numpy as np
# from dvclive import Live
import yaml
from dvclive import Live

n_estimators = yaml.safe_load(open("D:\exp-tracking-dvc\dvclive\params.yaml"))["n_estimators"]

data = pd.read_csv(r"C:\Users\u350272\OneDrive - WNS\Documents\StudyProj\mlops\experimenttracking\data\water_potability.csv")

from sklearn.model_selection import train_test_split
train_data,test_data = train_test_split(data,test_size=0.20,random_state=42)

def fill_missing_with_median(df):
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column].fillna(median_value,inplace=True)
    return df


# Fill missing values with median
train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

from sklearn.ensemble import RandomForestClassifier
import pickle
X_train = train_processed_data.iloc[:,0:-1].values
y_train = train_processed_data.iloc[:,-1].values

#n_estimators = 500

clf = RandomForestClassifier(n_estimators=n_estimators)
clf.fit(X_train,y_train)

# save 
pickle.dump(clf,open("model.pkl","wb"))


X_test = test_processed_data.iloc[:,0:-1].values
y_test = test_processed_data.iloc[:,-1].values

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

model = pickle.load(open('model.pkl',"rb"))

y_pred = model.predict(X_test)

acc = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
f1_score = f1_score(y_test,y_pred)


with Live(save_dvc_exp=True) as live:
    live.log_metric("accuracy_score",acc)
    live.log_metric("precision_score",precision)
    live.log_metric("recall_score",recall)
    live.log_metric("f1-score",f1_score)
    live.log_param("n_estimator", n_estimators)

# with Live(save_dvc_exp=True) as live:
#     live.log_metric("acc",acc)
#     live.log_metric("precision", precision)
#     live.log_metric("recall", recall)
#     live.log_metric("f1-score",f1_score)

#     live.log_param("n_estimators",n_estimators)