import pandas as pd
import matplotlib.pyplot as plt

def graphFrequencies(df, folder):
    dfnona=df.dropna()
    for col in dfnona.columns:
        val = dfnona[col].value_counts()
        val["NA"] = df[col].isna().sum()  # need to add column for NA manually
        gTitle = col + " Value Frequencies"
        val.plot(kind="bar", rot=15, title=gTitle, ylabel="Frequency", fontsize=5, figsize=(7.5, 5.5))
        fname = folder+"/" + col + ".png"
        plt.savefig(fname)

df=pd.read_csv("COVID-19_Case_Surveillance_Public_Use_Data_with_Geography.csv",parse_dates=["case_month"], infer_datetime_format=True, usecols=["case_month", "state_fips_code", "county_fips_code", "age_group", "sex", "race", "ethnicity", "case_positive_specimen_interval", "case_onset_interval", "process", "exposure_yn", "current_status", "symptom_status", "hosp_yn", "icu_yn", "death_yn", "underlying_conditions_yn"], dtype={"state_fips_code":"Int64", "county_fips_code":"Int64", "case_positive_specimen_interval":"Int64", "case_onset_interval":"Int64"})
print(df.shape)
df2=df.drop(["case_month"], axis=1)#I don't think we need case_month because the csv file only had data from 2020 but will keep for now just incase
# print(df["death_yn"].dropna().size)
graphFrequencies(df2, "Column_counts")

df["sex"].replace("Unknown", "Other/Unknown", inplace=True)
print(df["sex"].drop_duplicates())

df3=df.drop(["case_month","county_fips_code", "ethnicity", "case_positive_specimen_interval"], axis=1) #columns to definitely drop
print("Before: ", df3.shape)
df3.dropna(inplace=True)
df3bool=~df3.isin(["Unknown", "Missing"])#returns bool dataframe (false if unknown or missing)
df3b=df3bool.all(axis=1)#returns bool series (True if the row has all true values)
df3after=df3.loc[df3b]
print("After: ", df3after.shape)
df3after.to_csv("CovData/df3.csv")

df4=df.drop(["case_month","county_fips_code", "ethnicity", "case_positive_specimen_interval","underlying_conditions_yn", "case_onset_interval"], axis=1) #columns to definitely drop
print("Before: ", df4.shape)
df4.dropna(inplace=True)
df4bool=~df4.isin(["Unknown", "Missing"])#returns bool dataframe (false if unknown or missing)
df4b=df4bool.all(axis=1)#returns bool series (True if the row has all true values)
df4after=df4.loc[df4b]
print("After: ", df4after.shape)
df4after.to_csv("CovData/df4.csv")