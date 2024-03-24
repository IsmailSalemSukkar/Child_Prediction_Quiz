import joblib
import math

filename = "models/childPredictor.pkl"

loaded_model = joblib.load(filename)


# [
#    "wrkstat",
#    "age",
#    "female",
#    "relig",
#    "reliten",
#    "class_",
#    "chldidel",
#    "momWorks",
# ]

work = eval(input("Do you work? True or False"))
age = input("How old are you?")
gender = eval(input("Are you a girl? True or False"))
relig = eval(input("Are you part of a religion? True or False"))
religious = int(input("On a scale from 0-4, how religious are you?"))
social = int(
    input(
        "What social class are you, on a scale from 1-4? (1 is low class, 4 is high class)"
    )
)
idealChildren = int(input("How many children are in an ideal family?"))
momWorks = eval(input("Should a parent be a stay at home parent? True or False"))
result = loaded_model.predict(
    [[work, age, gender, relig, religious, social, idealChildren, momWorks]]
)

print(f"Between {math.floor(result[0])} and {math.ceil(result[0])}")
