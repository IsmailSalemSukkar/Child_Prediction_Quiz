import pandas as pd
import joblib


def clean_data(df):
    # Filter rows based on column: 'year'
    df = df[df["year"] == 2022]
    # Filter rows based on column: 'fepresch'
    df = df[
        (df["fepresch"] != ".i:  Inapplicable")
        & (df["fepresch"] != ".d:  Do not Know/Cannot Choose")
        & (df["fepresch"] != ".s:  Skipped on Web")
    ]
    # Replace missing values with "None" in column: 'relig'
    df = df.fillna({"relig": "None"})
    # Replace all instances of ".i:  Inapplicable" with "Weak" in column: 'reliten'
    df["reliten"] = df["reliten"].str.replace(
        ".i:  Inapplicable", "Weak", case=False, regex=False
    )
    # Replace all instances of "No religion" with "0" in column: 'reliten'
    df["reliten"] = df["reliten"].str.replace(
        "No religion", "0", case=False, regex=False
    )
    # Replace all instances of "weak" with "1" in column: 'reliten'
    df["reliten"] = df["reliten"].str.replace("weak", "1", case=False, regex=False)
    # Replace all instances of "Not very strong" with "2" in column: 'reliten'
    df["reliten"] = df["reliten"].str.replace(
        "Not very strong", "2", case=False, regex=False
    )
    # Replace all instances of "Somewhat strong (vol.)" with "3" in column: 'reliten'
    df["reliten"] = df["reliten"].str.replace(
        "Somewhat strong (vol.)", "3", case=False, regex=False
    )
    # Replace all instances of "Strong" with "4" in column: 'reliten'
    df["reliten"] = df["reliten"].str.replace("Strong", "4", case=False, regex=False)
    # Replace all instances of ".n:  No answer" with "None" in column: 'relig'
    df["relig"] = df["relig"].str.replace(
        ".n:  No answer", "None", case=False, regex=False
    )
    # Replace all instances of ".d:  Do not Know/Cannot Choose" with "None" in column: 'relig'
    df["relig"] = df["relig"].str.replace(
        ".d:  Do not Know/Cannot Choose", "None", case=False, regex=False
    )
    # Replace all instances of ".s:  Skipped on Web" with "None" in column: 'relig'
    df["relig"] = df["relig"].str.replace(
        ".s:  Skipped on Web", "None", case=False, regex=False
    )
    # Replace all instances of ".i:  Inapplicable" with "0" in column: 'hrs2'
    df["hrs2"] = df["hrs2"].str.replace(
        ".i:  Inapplicable", "0", case=False, regex=False
    )
    # Replace all instances of ".d:  Do not Know/Cannot Choose" with "0" in column: 'hrs2'
    df["hrs2"] = df["hrs2"].str.replace(
        ".d:  Do not Know/Cannot Choose", "0", case=False, regex=False
    )
    # Replace all instances of ".n:  No answer" with "0" in column: 'hrs2'
    df["hrs2"] = df["hrs2"].str.replace(".n:  No answer", "0", case=False, regex=False)
    # Replace all instances of ".s:  Skipped on Web" with "0" in column: 'hrs2'
    df["hrs2"] = df["hrs2"].str.replace(
        ".s:  Skipped on Web", "0", case=False, regex=False
    )
    # Replace all instances of ".d:  Do not Know/Cannot Choose" with "As many as you want" in column: 'chldidel'
    df["chldidel"] = df["chldidel"].str.replace(
        ".d:  Do not Know/Cannot Choose", "As many as you want", case=False, regex=False
    )
    # Replace all instances of ".i:  Inapplicable" with "As many as you want" in column: 'chldidel'
    df["chldidel"] = df["chldidel"].str.replace(
        ".i:  Inapplicable", "As many as you want", case=False, regex=False
    )
    # Filter rows based on column: 'year'
    df = df[df["year"] == 2022]
    # Replace all instances of "Working full time" with "1" in column: 'wrkstat'
    df["wrkstat"] = df["wrkstat"].str.replace(
        "Working full time", "1", case=False, regex=False
    )
    # Replace all instances of "Working part time" with "1" in column: 'wrkstat'
    df["wrkstat"] = df["wrkstat"].str.replace(
        "Working part time", "1", case=False, regex=False
    )
    # Replace all instances of "With a job, but not at work because of temporary illness, vacation, strike" with "1" in column: 'wrkstat'
    df["wrkstat"] = df["wrkstat"].str.replace(
        "With a job, but not at work because of temporary illness, vacation, strike",
        "1",
        case=False,
        regex=False,
    )
    # Replace all instances of "Unemployed, laid off, looking for work" with "1" in column: 'wrkstat'
    df["wrkstat"] = df["wrkstat"].str.replace(
        "Unemployed, laid off, looking for work", "1", case=False, regex=False
    )
    # Replace all instances of "In school" with "0" in column: 'wrkstat'
    df["wrkstat"] = df["wrkstat"].str.replace("In school", "0", case=False, regex=False)
    # Replace all instances of "Keeping house" with "0" in column: 'wrkstat'
    df["wrkstat"] = df["wrkstat"].str.replace(
        "Keeping house", "0", case=False, regex=False
    )
    # Replace all instances of "Other" with "0" in column: 'wrkstat'
    df["wrkstat"] = df["wrkstat"].str.replace("Other", "0", case=False, regex=False)
    # Filter rows based on column: 'wrkstat'
    df = df[df["wrkstat"] != "Retired"]
    # Replace all instances of ".s: Skipped on Web" with "0" in column: 'wrkstat'
    df["wrkstat"] = df["wrkstat"].str.replace(
        ".s: Skipped on Web", "0", case=False, regex=False
    )
    # Replace all instances of ".s: Skipped on Web" with "0" in column: 'wrkstat'
    df["wrkstat"] = df["wrkstat"].str.replace(
        ".s:  Skipped on Web", "0", case=False, regex=False
    )

    df["wrkstat"] = df["wrkstat"].str.replace(
        ".n:  No answer", "0", case=False, regex=False
    )

    # Replace all instances of ".d: Do not Know/Cannot Choose" with "0" in column: 'wrkstat'
    df["wrkstat"] = df["wrkstat"].str.replace(
        ".d: Do not Know/Cannot Choose", "0", case=False, regex=False
    )
    # Replace all instances of ".n: No answer" with "0" in column: 'wrkstat'
    df["wrkstat"] = df["wrkstat"].str.replace(
        ".n: No answer", "0", case=False, regex=False
    )

    df["chldidel"] = df["chldidel"].str.replace(
        "As many as you want", "8", case=False, regex=False
    )
    # Replace all instances of "7 or more" with "7" in column: 'chldidel'
    df["chldidel"] = df["chldidel"].str.replace(
        "7 or more", "7", case=False, regex=False
    )
    # Replace all instances of ".d:  Do not Know/Cannot Choose" with "0" in column: 'reliten'
    df["reliten"] = df["reliten"].str.replace(
        ".d:  Do not Know/Cannot Choose", "0", case=False, regex=False
    )
    # Change column type to int64 for column: 'reliten'
    df = df.astype({"reliten": "int64"})

    df = df[
        (df["class_"].str.contains("class", regex=False, na=False))
        & (df["class_"] != "No class")
    ]
    # Replace all instances of "Lower class" with "0" in column: 'class_'
    df["class_"] = df["class_"].str.replace("Lower class", "0", case=False, regex=False)
    # Replace all instances of "Working Class" with "1" in column: 'class_'
    df["class_"] = df["class_"].str.replace(
        "Working Class", "1", case=False, regex=False
    )
    # Replace all instances of "Middle Class" with "2" in column: 'class_'
    df["class_"] = df["class_"].str.replace(
        "Middle Class", "2", case=False, regex=False
    )
    # Replace all instances of "Upper class" with "3" in column: 'class_'
    df["class_"] = df["class_"].str.replace("Upper class", "3", case=False, regex=False)
    # Change column type to int64 for column: 'class_'
    df = df.astype({"class_": "int64"})
    # Change column type to int64 for column: 'chldidel'
    df = df.astype({"chldidel": "int64"})
    # Filter rows based on column: 'chldidel'
    df = df[df["chldidel"] != 8]
    # Filter rows based on column: 'age'
    df = df[df["age"] != ".n:  No answer"]
    # Replace all instances of "89 or older" with "89" in column: 'age'
    df["age"] = df["age"].str.replace("89 or older", "89", case=False, regex=False)
    # Change column type to int64 for column: 'age'
    df = df.astype({"age": "int64"})
    # Replace all instances of "8 or more" with "8" in column: 'childs'
    df["childs"] = df["childs"].str.replace("8 or more", "8", case=False, regex=False)
    # Replace all instances of ".i:  Inapplicable" with "0" in column: 'childs'
    df["childs"] = df["childs"].str.replace(
        ".i:  Inapplicable", "0", case=False, regex=False
    )
    # Change column type to int64 for column: 'childs'
    df = df.astype({"childs": "int64"})
    # Drop column: 'hrs2'
    df = df.drop(columns=["hrs2"])
    # Change column type to int64 for column: 'wrkstat'
    df = df.astype({"wrkstat": "int64"})
    # Change column type to bool for column: 'wrkstat'
    df = df.astype({"wrkstat": "bool"})
    # Rename column 'sex' to 'female'
    df = df.rename(columns={"sex": "female"})
    # Replace all instances of "female" with "1" in column: 'female'
    df["female"] = df["female"].str.replace("female", "1", case=False, regex=False)
    # Replace all instances of "male" with "0" in column: 'female'
    df["female"] = df["female"].str.replace("male", "0", case=False, regex=False)
    # Filter rows based on column: 'female'
    df = df[(df["female"] == "0") | (df["female"] == "1")]
    # Change column type to int64 for column: 'female'
    df = df.astype({"female": "int64"})
    # Change column type to bool for column: 'female'
    df = df.astype({"female": "bool"})
    # Replace all instances of "AGREE" with "1" in column: 'fechld'
    df.loc[df["fechld"].str.lower() == "AGREE".lower(), "fechld"] = "1"
    # Replace all instances of "strongly agree" with "1" in column: 'fechld'
    df["fechld"] = df["fechld"].str.replace(
        "strongly agree", "1", case=False, regex=False
    )
    # Rename column 'fechld' to 'momWorks'
    df = df.rename(columns={"fechld": "momWorks"})
    # Replace all instances of "disagree" with "0" in column: 'momWorks'
    df["momWorks"] = df["momWorks"].str.replace(
        "disagree", "0", case=False, regex=False
    )
    # Replace all instances of "strongly 0" with "0" in column: 'momWorks'
    df["momWorks"] = df["momWorks"].str.replace(
        "strongly 0", "0", case=False, regex=False
    )
    # Filter rows based on column: 'momWorks'
    df = df[(df["momWorks"] == "0") | (df["momWorks"] == "1")]
    # Change column type to int64 for column: 'momWorks'
    df = df.astype({"momWorks": "int64"})
    # Change column type to bool for column: 'momWorks'
    df = df.astype({"momWorks": "bool"})
    # Drop columns: 'attend', 'denom' and 4 other columns
    df = df.drop(columns=["attend", "denom", "fund", "pillok", "fepresch", "ballot"])
    # Replace all instances of "None" with "0" in column: 'relig'
    df["relig"] = df["relig"].str.replace("None", "0", case=False, regex=False)
    # Replace all instances of "\\b[1-9a-zA-Z]+\\b" with "1" in column: 'relig'
    df["relig"] = df["relig"].str.replace(
        "\\b[1-9a-zA-Z]+\\b", "1", case=False, regex=True
    )
    # Replace all instances of "1-1" with "1" in column: 'relig'
    df["relig"] = df["relig"].str.replace("1-1", "1", case=False, regex=False)
    # Replace all instances of "1/1" with "1" in column: 'relig'
    df["relig"] = df["relig"].str.replace("1/1", "1", case=False, regex=False)
    # Replace all instances of "1 1 1" with "1" in column: 'relig'
    df["relig"] = df["relig"].str.replace("1 1 1", "1", case=False, regex=False)
    # Change column type to int64 for column: 'relig'
    df = df.astype({"relig": "int64"})
    # Change column type to bool for column: 'relig'
    df = df.astype({"relig": "bool"})
    return df


# Loaded variable 'df' from URI: /home/izzy/Documents/CodingProjects/NORCWebApp/GSS.csv
df = pd.read_csv(r"/home/izzy/Documents/CodingProjects/NORCWebApp/GSS2.csv")

df_clean = clean_data(df.copy())

from sklearn.model_selection import train_test_split

children, indVar = (
    df_clean["childs"].values,
    df_clean[
        [
            "wrkstat",
            "age",
            "female",
            "relig",
            "reliten",
            "class_",
            "chldidel",
            "momWorks",
        ]
    ].values,
)

children_train, children_test, ind_train, ind_test = train_test_split(
    children, indVar, test_size=0.30, random_state=0
)

list(df_clean.columns)

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

model = GradientBoostingRegressor().fit(ind_train, children_train)
print(model)

import numpy as np

predictions = model.predict(ind_test)
mse = mean_squared_error(children_test, predictions)
print("MSE:", mse)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
r2 = r2_score(children_test, predictions)
print(r2)

# Save the model as a pickle file
filename = "./childPredictor.pkl"
joblib.dump(model, filename)
