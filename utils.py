import pandas as pd

SERVICE_COLS = ["PhoneService", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

def load_clean(path="data/Telco-Customer-Churn.csv"):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")  # blanks -> NaN
    df = df.drop(columns="customerID").drop_duplicates()                     # duplicates remove
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df

def engineer(df):
    df = df.copy()
    df["tenure_group"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72],
                                labels=["0-12", "13-24", "25-48", "49-72"]).astype(str)
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["num_services"] = sum((df[c] == "Yes").astype(int) for c in SERVICE_COLS)
    return df