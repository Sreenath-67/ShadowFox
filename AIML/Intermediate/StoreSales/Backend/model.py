from sklearn.linear_model import LinearRegression

def train_model(df):
    X = df[["Sales"]]
    y = df["Profit"]

    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_profit(model, data):
    return float(model.predict([[data["sales"]]])[0])