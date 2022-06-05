import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.preprocessing import LabelEncoder

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
#graphFrequencies(df2, "Column_counts")
#####selectkbest categorical
dfxnum=df.drop(["case_month", "state_fips_code", "county_fips_code", "age_group", "sex", "race", "ethnicity", "process", "exposure_yn", "current_status", "symptom_status", "hosp_yn", "icu_yn", "underlying_conditions_yn"], axis=1)
print(dfxnum["case_positive_specimen_interval"].drop_duplicates())
print(dfxnum["case_onset_interval"].drop_duplicates())
dfxnum=dfxnum.dropna()
print(dfxnum["case_positive_specimen_interval"].drop_duplicates())
print(dfxnum["case_onset_interval"].drop_duplicates())
dfxnumbool=~dfxnum.isin(["Unknown", "Missing"])#returns bool dataframe (false if unknown or missing)
dfxbnum=dfxnumbool.all(axis=1)#returns bool series (True if the row has all true values)
dfx2num=dfxnum.loc[dfxbnum]
#featuresnum=["case_positive_specimen_interval", "case_onset_interval"]
print(dfx2num["case_positive_specimen_interval"].drop_duplicates())
print(dfx2num["case_onset_interval"].drop_duplicates())
featuresnum=pd.Series(dfx2num.drop("death_yn",axis=1).columns)
xnum=dfx2num.drop("death_yn",axis=1).values
ynum=dfx2num["death_yn"].values
selectnum = SelectKBest(score_func=f_classif, k=1)
znum = selectnum.fit_transform(xnum, ynum)
colsnum = selectnum.get_support()
print("K=1", " Features: ", featuresnum.loc[colsnum].values)
####################
dfx=df.drop(["case_month","case_positive_specimen_interval", "case_onset_interval"], axis=1)
dfx["sex"].replace("Unknown", "Other/Unknown", inplace=True)
dfx=dfx.dropna()
dfxbool=~dfx.isin(["Unknown", "Missing"])#returns bool dataframe (false if unknown or missing)
dfxb=dfxbool.all(axis=1)#returns bool series (True if the row has all true values)
dfx2=dfx.loc[dfxb]
print(dfx2["sex"].drop_duplicates())
dfx3=dfx2.copy()
for col in ["age_group", "sex", "race", "ethnicity", "process", "exposure_yn", "current_status", "symptom_status", "hosp_yn", "icu_yn", "death_yn", "underlying_conditions_yn"]:
    le=LabelEncoder()
    dfx2.loc[:,col]=le.fit_transform(dfx2.loc[:,col])

features=pd.Series(dfx2.drop(["death_yn"],axis=1).columns)
x=dfx2.drop(["death_yn"],axis=1).values
y=dfx2.loc[:,"death_yn"].values
for kval in range(1,14):
    select = SelectKBest(chi2, k=kval)
    z = select.fit_transform(x, y)
    print(select.scores_)
    cols = select.get_support()
    print("K=", kval, " Features: ", features.loc[cols].values)
##########################
#graphFrequencies(df2, "Column_counts")
df["sex"].replace("Unknown", "Other/Unknown", inplace=True)
print(df["sex"].drop_duplicates())
print("Exposure: ", df["exposure_yn"].drop_duplicates())
df3=df.drop(["case_month","county_fips_code", "ethnicity", "case_positive_specimen_interval"], axis=1) #columns to definitely drop
print("Before: ", df3.shape)
df3.dropna(inplace=True)
print("Exposure: ", df3["exposure_yn"].drop_duplicates())
df3bool=~df3.isin(["Unknown", "Missing"])#returns bool dataframe (false if unknown or missing)
df3b=df3bool.all(axis=1)#returns bool series (True if the row has all true values)
df3after=df3.loc[df3b]
print("Exposure: ", df3after["exposure_yn"].drop_duplicates())
print("After: ", df3after.shape)
df3after.to_csv("CovData/df3.csv")

df4=df.drop(["case_month","county_fips_code", "ethnicity", "case_positive_specimen_interval","underlying_conditions_yn", "case_onset_interval"], axis=1) #columns to definitely drop
print("Before: ", df4.shape)
df4.dropna(inplace=True)
df4bool=~df4.isin(["Unknown", "Missing"])#returns bool dataframe (false if unknown or missing)
df4b=df4bool.all(axis=1)#returns bool series (True if the row has all true values)
df4after=df4.loc[df4b]
print("Exposure4: ", df4after["exposure_yn"].drop_duplicates())
print("After: ", df4after.shape)
df4after.to_csv("CovData/df4.csv")