import pandas as pd

path = "D:/workspace/learn AI/machine_learning/titanic/dataset/train.csv"
data = pd.read_csv(path)

data["Name"] = data["Name"].str.extract(r', ([^.]+)\.')
term_data = data.dropna()
mapping = {
    "Mr": term_data[term_data["Name"] == "Mr"]["Age"].sum()/len(term_data[term_data["Name"] == "Mr"]),
    "Mrs": term_data[term_data["Name"] == "Mrs"]["Age"].sum()/len(term_data[term_data["Name"] == "Mrs"]),
    "Miss": term_data[term_data["Name"] == "Miss"]["Age"].sum()/len(term_data[term_data["Name"] == "Miss"]),
    "Master": term_data[term_data["Name"] == "Master"]["Age"].sum()/len(term_data[term_data["Name"] == "Master"]),
    "Dr": term_data[term_data["Name"] == "Dr"]["Age"].sum()/len(term_data[term_data["Name"] == "Dr"])
}

data["Age"] = data["Age"].fillna(data["Name"].map(mapping))
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
data["isAlone"] = (data["FamilySize"] == 1).astype(int)
data = data.drop(columns=["Ticket", "Cabin"])

# submission = pd.DataFrame({
#     "PassengerId": submit_test_df["PassengerId"],
#     "Survived": submit
# })
data.to_csv("processed_train2.csv", index=False)

data.isna().to_csv("is_na.csv", index=False)

path2 = "D:/workspace/learn AI/machine_learning/titanic/dataset/test.csv"
test_data = pd.read_csv(path2)


