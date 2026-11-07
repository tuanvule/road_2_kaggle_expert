import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

def processed_data(data):
    data["Name"] = data["Name"].str.extract(r', ([^.]+)\.')
    term_data = data.dropna()
    mapping = {
        "Mr": term_data[term_data["Name"] == "Mr"]["Age"].sum()/len(term_data[term_data["Name"] == "Mr"]),
        "Mrs": term_data[term_data["Name"] == "Mrs"]["Age"].sum()/len(term_data[term_data["Name"] == "Mrs"]),
        "Miss": term_data[term_data["Name"] == "Miss"]["Age"].sum()/len(term_data[term_data["Name"] == "Miss"]),
        "Master": term_data[term_data["Name"] == "Master"]["Age"].sum()/len(term_data[term_data["Name"] == "Master"]),
        "Dr": 0 if len(term_data[term_data["Name"] == "Dr"]) < 1 else term_data[term_data["Name"] == "Dr"]["Age"].sum()/len(term_data[term_data["Name"] == "Dr"])
    }

    data["Age"] = data["Age"].fillna(data["Name"].map(mapping))
    data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    data["isAlone"] = (data["FamilySize"] == 1).astype(int)
    data = data.drop(columns=["Ticket", "Cabin"])
    return data

def Entrophy(current_df):
  if len(current_df) == 0: return "empty"
  p1 = (current_df["Survived"] == 1).sum() / len(current_df)
  p2 = (current_df["Survived"] == 0).sum() / len(current_df)
  if p1 == 0 or p2 == 0:
      return 1
  return -(p1 * np.log2(p1) + p2 * np.log2(p2))

def childrentNumberEntrophy(left, right):
  df = pd.concat([left, right])
  # print(len(left), len(right), len(df))
  entrophy_left = len(left)/len(df) * Entrophy(left)
  entrophy_right = len(right)/len(df) * Entrophy(right)
  return entrophy_left + entrophy_right

def childrenCategoryEntrophy(df, head, cats):
    sum = 0
    for cat in cats:
        term_df = df[df[head] == cat]
        sum += len(term_df) / len(df) * Entrophy(term_df)
    return sum


def bestFeature(current_df):

    parrent_entrophy = Entrophy(current_df)

    best_feature = ""
    best_threshold = -1
    best_imformation_gain = 0
    feature_type = "n"
    cats = []

    ig = []

    def checkNumberFeature():
        nonlocal best_feature, best_threshold, best_imformation_gain
        numeric_feature = current_df.select_dtypes(include="number").drop(columns=["PassengerId"])
        heads = numeric_feature.columns
        for head in heads:
            if head == "Survived": continue
            df = numeric_feature[["Survived",head]]
            values = np.sort(np.unique(df[head]))
            thresholds = (values[:-1] + values[1:]) / 2
            for t in thresholds:
                df_left = df[df[head] <= t]
                df_right = df[df[head] > t]

                if len(df_left) == 0 or len(df_right) == 0:
                    continue
                imfomation_gain = parrent_entrophy - childrentNumberEntrophy(df_left, df_right)
                ig.append(("num", head, thresholds, imfomation_gain))
                if best_imformation_gain < imfomation_gain:
                    best_imformation_gain = imfomation_gain
                    best_feature = head
                    best_threshold = t
                    # print(best_feature, best_imformation_gain ,best_threshold)

    def checkCategoryFeature():
        nonlocal best_feature, best_threshold, best_imformation_gain, feature_type, cats
        cat_feature = current_df[["Survived", "Sex", "Embarked"]]
        heads = cat_feature.columns
        for head in heads:
            if head == "Survived": continue
            df = current_df[["Survived",head]]
            # print(current_df[head].unique())
            loc_cats = current_df[head].unique()

            imformation_gain = parrent_entrophy - childrenCategoryEntrophy(df, head, loc_cats)
            ig.append(("cats", head, imformation_gain))
            if best_imformation_gain < imformation_gain:
                best_imformation_gain = imformation_gain
                best_feature = head
                feature_type = "c"
                cats = loc_cats


    checkNumberFeature()
    checkCategoryFeature()
    # if best_feature == "":
    #     print("EMPTY ___ ", current_df)
    #     print("", parrent_entrophy, len(current_df))
    #     print("1 len: ", (current_df["Survived"] == 1).sum())
    #     print("0 len: ", (current_df["Survived"] == 0).sum())
    #     print(ig)

    return (best_feature, best_imformation_gain ,best_threshold, cats, feature_type)


def buildThree(df, max_dept):
    tree = []
    count = 1

    def buildNodes(current_df, dept, c):
        # print("howmany")
        nonlocal count

        if dept >= max_dept or len(current_df) <= 1 or Entrophy(current_df) == 0:
            tree.append((c, "end", ((current_df["Survived"] == 1).sum() >= (current_df["Survived"] == 0).sum()).astype(int), "", [], Entrophy(current_df), dept))
            return c

        best_feature, best_imformation_gain ,best_threshold, loc_cats, feature_type = bestFeature(current_df)
        if best_feature == "":

            tree.append((c, "end_wthhou", ((current_df["Survived"] == 1).sum() >= (current_df["Survived"] == 0).sum()).astype(int), "", [], Entrophy(current_df), dept))
            return

        if feature_type == "n":
            left = current_df[current_df[best_feature] >= best_threshold]
            right = current_df[current_df[best_feature] < best_threshold]
            child_node_indexs = []
            if not left.empty:
                count += 1
                left_child = count
                buildNodes(left, dept + 1, left_child)
                child_node_indexs.append(left_child)
            if not right.empty:
                count += 1
                right_child = count
                buildNodes(right, dept + 1, right_child)
                child_node_indexs.append(right_child)
            tree.append((c, feature_type, best_feature, loc_cats if feature_type == "c" else best_threshold, child_node_indexs, Entrophy(current_df), dept))

        elif feature_type == "c":
            child_node_indexs = []

            for cat in loc_cats:
                term_df = current_df[current_df[best_feature] == cat]

                if not term_df.empty:
                    count += 1
                    next_child = count
                    buildNodes(term_df, dept + 1, next_child)
                    child_node_indexs.append(next_child)

            tree.append((c, feature_type, best_feature, loc_cats if feature_type == "c" else best_threshold, child_node_indexs, Entrophy(current_df), dept))

        return c

    buildNodes(df, 1, 1)

    return tree


def print_tree(nodes, node_id=1, indent="", branch="ROOT"):
    # Tìm node theo ID
    node = next((n for n in nodes if n[0] == node_id), None)

    if node is None:
        return

    c, feature_type, best_feature, condition, children, entropy, depth = node

    # In thông tin node
    if feature_type == "end":
        print(
            f"{indent}└── [{branch}] "
            f"Leaf #{c}: Predict={best_feature}, "
            f"Entropy={entropy:.4f}, Depth={depth}"
        )
        return

    print(
        f"{indent}└── [{branch}] "
        f"Node #{c}: {best_feature}, "
        f"Condition={condition}, "
        f"Entropy={entropy:.4f}, Depth={depth}"
    )

    # In các node con
    if feature_type == "n":
        left_condition = f"{best_feature} <= {condition}"
        right_condition = f"{best_feature} > {condition}"

    elif feature_type == "c":
        categories = condition
        left_condition = f"{best_feature} == {categories[0]}"
        right_condition = f"{best_feature} == {categories[1]}"

    for i, child_id in enumerate(children):
        if i == 0:
            child_branch = left_condition
        else:
            child_branch = right_condition

        print_tree(
            nodes,
            child_id,
            indent + "    ",
            child_branch
        )


def decision_tree(test, tree, node_id=1):
    node = next((n for n in tree if n[0] == node_id), None)

    if node == None:
        return


    # (6, 'end', np.int64(0), '', [], np.float64(0.5759084429149788), 5)
    # (4, 'n', 'Age', np.float64(32.25), [5, 6], np.float64(0.49620057900956627), 4)
    # (1, 'c', 'Sex', array(['male', 'female'], dtype=object), [2, 17], np.float64(0.9727492758110255), 1)
    if node[1] == 'n':
        return decision_tree(test, tree, node_id = node[4][0] if test[node[2]] >= node[3] else node[4][1])
    elif node[1] == 'c':
        for i, t in enumerate(node[3]):
            if test[node[2]] == t:
                return decision_tree(test, tree, node_id = node[4][i])

    else: return node[2]

def classified(tests, tree):

    submit = []

    for index, test in tests.iterrows():
        submit.append(decision_tree(test, tree))
    # print(submit)
    # print((submit == tests["Survived"]).head(10))
    # print((submit == tests["Survived"]).sum()/len(submit))
    return submit



df = pd.read_csv('dataset/train.csv')


train_df, test_df = train_test_split(
    df,
    test_size=0.1,
    random_state=42,
    shuffle=True
)

train_df = processed_data(train_df)
test_df = processed_data(test_df)

submit_test_df = pd.read_csv('dataset/test.csv')
submit_test_df = processed_data(submit_test_df)


submit = classified(submit_test_df, tree)
print(submit)
submission = pd.DataFrame({
    "PassengerId": submit_test_df["PassengerId"],
    "Survived": submit
})
submission.to_csv("submission.csv", index=False)
print(len(test_df))


