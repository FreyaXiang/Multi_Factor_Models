import numpy as np
import pandas as pd
import os
import glob
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# [[300122, 0.56], [111111, 0.67]...]
def getFileInfo(fileName):
    file = open(fileName, "r")
    dataVecs = [line.strip().split(",") for line in file.readlines()[1:]]
    return dataVecs

def spearmanCalc():
    # get the info from main chart
    file = open("factors.txt", "r", encoding="utf-8")
    dataSet = [line.strip().split() for line in file.readlines()]
    stockNum = [vec[0] for vec in dataSet]
    scoreVec = [vec[2] for vec in dataSet]

    finalSet = {"score": scoreVec}
    count = 1
    # get all tht file names from the folder SingleFactor
    path = "./SingleFactor"
    for fileName in glob.glob(os.path.join(path, "*.csv")):
        with open(os.path.join(os.getcwd(), fileName), "r") as f:
            dataVecs = getFileInfo(fileName)
            vecToBeAdd = []
            for num in stockNum:
                for vec in dataVecs:
                    if num == vec[0]:
                        if (vec[1] == ""):
                            vecToBeAdd.append(np.nan)
                        else:
                            vecToBeAdd.append(vec[1])
                        break
            finalSet[count] = vecToBeAdd
            count += 1

    # calculate
    finalSet = pd.DataFrame(finalSet)
    finalSet2 = finalSet.astype(float)
    result = finalSet2.corr(method = "spearman")
    # pd.set_option('display.expand_frame_repr', False)
    result = result.sort_values(by = ['score'])
    print(result.iloc[-7:-2,0])
    # get the highest 5 factors: 47 4 48 26 34
    return finalSet2.loc[:, [47, 4, 48, 26, 34, 'score']]

def regression():
    data = spearmanCalc()
    # Independent and dependent vars
    X = pd.DataFrame(data.iloc[:, :-1])
    y = pd.DataFrame(data.iloc[:, -1])
    print(X)
    poly_features = PolynomialFeatures(degree=3)
    X_poly = poly_features.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)
    y_poly_pred = model.predict(X_poly)

    # root mean square error and R square
    rmse = np.sqrt(mean_squared_error(y, y_poly_pred))
    # r_sq = r2_score(y, y_poly_pred)
    r_sq = model.score(X_poly, y)
    print('Root mean-square error:', rmse)
    print('coefficient of determination(𝑅²) :', r_sq)
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)
    print("Predicted response:", y_poly_pred)

    # Calculate the spearman coefficient of real values and predicted values
    result = y
    result['predict'] = y_poly_pred
    result = result.astype(float)
    coefficient = result.corr(method='spearman')
    print(coefficient)

regression()

