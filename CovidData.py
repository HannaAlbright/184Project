import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.preprocessing import LabelEncoder

def graphFrequencies(df, folder):
    dfnona=df.dropna()
    for col in dfnona.columns:
        val = dfnona[col].value_counts() #counts frequency of all values except NA
        val["NA"] = df[col].isna().sum()  # need to add column for NA manually
        gTitle = col + " Value Frequencies"
        val.plot(kind="bar", rot=110, title=gTitle, ylabel="Frequency", fontsize=10, figsize=(7.10, 10.10))
        fname = folder+"/" + col + ".png"
        plt.savefig(fname)

df=pd.read_csv("COVID-19_Case_Surveillance_Public_Use_Data_with_Geography.csv",parse_dates=["case_month"], infer_datetime_format=True, usecols=["case_month", "state_fips_code", "county_fips_code", "age_group", "sex", "race", "ethnicity", "case_positive_specimen_interval", "case_onset_interval", "process", "exposure_yn", "current_status", "symptom_status", "hosp_yn", "icu_yn", "death_yn", "underlying_conditions_yn"], dtype={"state_fips_code":"Int64", "county_fips_code":"Int64", "case_positive_specimen_interval":"Int64", "case_onset_interval":"Int64"})
print(df.shape)
df2=df.drop(["case_month"], axis=1)#I don't think we need case_month because the csv file only had data from 2020 but will keep for now just incase
#graphFrequencies(df2, "Column_counts")
#####selectkbest categorical
dfxnum=df.drop(["case_month", "state_fips_code", "county_fips_code", "age_group", "sex", "race", "ethnicity", "process", "exposure_yn", "current_status", "symptom_status", "hosp_yn", "icu_yn", "underlying_conditions_yn"], axis=1) #drop categorical values
# print(dfxnum["case_positive_specimen_interval"].drop_duplicates())
# print(dfxnum["case_onset_interval"].drop_duplicates())
dfxnum=dfxnum.dropna()
# print(dfxnum["case_positive_specimen_interval"].drop_duplicates())
# print(dfxnum["case_onset_interval"].drop_duplicates())
dfxnumbool=~dfxnum.isin(["Unknown", "Missing"])#returns bool dataframe (false if unknown or missing)
dfxbnum=dfxnumbool.all(axis=1)#returns bool series (True if the row has all true values)
dfx2num=dfxnum.loc[dfxbnum]
#featuresnum=["case_positive_specimen_interval", "case_onset_interval"]
# print(dfx2num["case_positive_specimen_interval"].drop_duplicates())
# print(dfx2num["case_onset_interval"].drop_duplicates())
featuresnum=pd.Series(dfx2num.drop("death_yn",axis=1).columns) #series of the numerical features
xnum=dfx2num.drop("death_yn",axis=1).values #array of x values of the numerical features
ynum=dfx2num["death_yn"].values  #array of y values (target variable) which is death_yn
selectnum = SelectKBest(score_func=f_classif, k=1) #since there are only 2 numerical features, we only need k=1 to see which is the better feature
znum = selectnum.fit_transform(xnum, ynum)
print(selectnum.scores_)
colsnum = selectnum.get_support()
print("K=1", " Features: ", featuresnum.loc[colsnum].values)
####################
dfx=df.drop(["case_month","case_positive_specimen_interval", "case_onset_interval"], axis=1)
dfx["sex"].replace("Unknown", "Other/Unknown", inplace=True) #replace unknown with other/unknown as a third option for gender and so that it isn't dropped
dfx=dfx.dropna()
dfxbool=~dfx.isin(["Unknown", "Missing"])#returns bool dataframe (false if unknown or missing)
dfxb=dfxbool.all(axis=1)#returns bool series (True if the row has all true values)
dfx2=dfx.loc[dfxb] #get only the rows without unknown or missing
# print(dfx2["sex"].drop_duplicates())
for col in ["age_group", "sex", "race", "ethnicity", "process", "exposure_yn", "current_status", "symptom_status", "hosp_yn", "icu_yn", "death_yn", "underlying_conditions_yn"]:
    le=LabelEncoder()#label encode categorical variables to prepare for selectkbest
    dfx2.loc[:,col]=le.fit_transform(dfx2.loc[:,col])

# features=pd.Series(dfx2.drop(["death_yn"],axis=1).columns)
# x=dfx2.drop(["death_yn"],axis=1).values
# y=dfx2.loc[:,"death_yn"].values
# for kval in range(1,14):
#     select = SelectKBest(chi2, k=kval)
#     z = select.fit_transform(x, y)
#     print(select.scores_)
#     cols = select.get_support()
#     print("K=", kval, " Features: ", features.loc[cols].values)
##########################
#graphFrequencies(df2, "Column_counts")
df["sex"].replace("Unknown", "Other/Unknown", inplace=True)
# print(df["sex"].drop_duplicates())
# print("Exposure: ", df["exposure_yn"].drop_duplicates())
# df3=df.drop(["case_month","county_fips_code", "ethnicity", "case_positive_specimen_interval"], axis=1) #columns to definitely drop
# print("Before: ", df3.shape)
# df3.dropna(inplace=True)
# print("Exposure: ", df3["exposure_yn"].drop_duplicates())
# df3bool=~df3.isin(["Unknown", "Missing"])#returns bool dataframe (false if unknown or missing)
# df3b=df3bool.all(axis=1)#returns bool series (True if the row has all true values)
# df3after=df3.loc[df3b]
# print("Exposure: ", df3after["exposure_yn"].drop_duplicates())
# print("After: ", df3after.shape)
# df3after.to_csv("CovData/df3.csv", index=False)
# ['icu_yn', 'hosp_yn', 'age_group,'county_fips_code', 'process', 'underlying_conditions_yn', 'ethnicity,  'sex', 'symptom_status, 'race', 'current_status', ‘'state_fips_code', 'exposure_yn'


df1=df.drop(["case_month","underlying_conditions_yn", "ethnicity", "sex","symptom_status", "race", "current_status", "state_fips_code", "exposure_yn","case_onset_interval", "case_positive_specimen_interval","county_fips_code"], axis=1) #columns to definitely drop
print("Before: ", df1.shape)
df1.dropna(inplace=True)
df1bool=~df1.isin(["Unknown", "Missing"])#returns bool dataframe (false if unknown or missing)
df1b=df1bool.all(axis=1)#returns bool series (True if the row has all true values)
df1after=df1.loc[df1b]
print("After: ", df1after.shape)
df1after.to_csv("CovData/df1.csv", index=False)


df2=df.drop(["case_month", "ethnicity",  "sex","symptom_status", "race", "current_status", "state_fips_code", "exposure_yn","case_onset_interval", "case_positive_specimen_interval","county_fips_code"], axis=1) #columns to definitely drop
print("Before: ", df2.shape)
df2.dropna(inplace=True)
df2bool=~df2.isin(["Unknown", "Missing"])#returns bool dataframe (false if unknown or missing)
df2b=df2bool.all(axis=1)#returns bool series (True if the row has all true values)
df2after=df2.loc[df2b]
print("After: ", df2after.shape)
df2after.to_csv("CovData/df2.csv", index=False)
#'icu_yn', 'hosp_yn', 'age_group,'county_fips_code', 'process',  'underlying_conditions_yn', 'ethnicity,  'sex', 'symptom_status, 'race', 'current_status', ‘'state_fips_code', 'exposure_yn'

df3=df.drop(["case_month", "sex","symptom_status", "race", "current_status", "state_fips_code", "exposure_yn","case_onset_interval", "case_positive_specimen_interval","county_fips_code"], axis=1) #columns to definitely drop
print("Before: ", df3.shape)
df3.dropna(inplace=True)
df3bool=~df3.isin(["Unknown", "Missing"])#returns bool dataframe (false if unknown or missing)
df3b=df3bool.all(axis=1)#returns bool series (True if the row has all true values)
df3after=df3.loc[df3b]
print("After: ", df3after.shape)
df3after.to_csv("CovData/df3.csv", index=False)
#'icu_yn', 'hosp_yn', 'age_group,'county_fips_code', 'process',  'underlying_conditions_yn', 'ethnicity,  'sex', 'symptom_status, 'race', 'current_status', ‘'state_fips_code', 'exposure_yn'

df4=df.drop(["case_month","symptom_status", "race", "current_status", "state_fips_code", "exposure_yn","case_onset_interval", "case_positive_specimen_interval","county_fips_code"], axis=1) #columns to definitely drop
print("Before: ", df4.shape)
df4.dropna(inplace=True)
df4bool=~df4.isin(["Unknown", "Missing"])#returns bool dataframe (false if unknown or missing)
df4b=df4bool.all(axis=1)#returns bool series (True if the row has all true values)
df4after=df4.loc[df4b]
print("After: ", df4after.shape)
df4after.to_csv("CovData/df4.csv", index=False)
#'icu_yn', 'hosp_yn', 'age_group,'county_fips_code', 'process',  'underlying_conditions_yn', 'ethnicity,  'sex', 'symptom_status, 'race', 'current_status', ‘'state_fips_code', 'exposure_yn'

df5=df.drop(["case_month", "race", "current_status", "state_fips_code", "exposure_yn","case_onset_interval", "case_positive_specimen_interval","county_fips_code"], axis=1) #columns to definitely drop
print("Before: ", df5.shape)
df5.dropna(inplace=True)
df5bool=~df5.isin(["Unknown", "Missing"])#returns bool dataframe (false if unknown or missing)
df5b=df5bool.all(axis=1)#returns bool series (True if the row has all true values)
df5after=df5.loc[df5b]
print("After: ", df5after.shape)
df5after.to_csv("CovData/df5.csv", index=False)
#'icu_yn', 'hosp_yn', 'age_group,'county_fips_code', 'process',  'underlying_conditions_yn', 'ethnicity,  'sex', 'symptom_status, 'race', 'current_status', ‘'state_fips_code', 'exposure_yn'

df6=df.drop(["case_month","current_status", "state_fips_code", "exposure_yn","case_onset_interval", "case_positive_specimen_interval","county_fips_code"], axis=1) #columns to definitely drop
print("Before: ", df6.shape)
df6.dropna(inplace=True)
df6bool=~df6.isin(["Unknown", "Missing"])#returns bool dataframe (false if unknown or missing)
df6b=df6bool.all(axis=1)#returns bool series (True if the row has all true values)
df6after=df6.loc[df6b]
print("After: ", df6after.shape)
df6after.to_csv("CovData/df6.csv", index=False)
#'icu_yn', 'hosp_yn', 'age_group,'county_fips_code', 'process',  'underlying_conditions_yn', 'ethnicity,  'sex', 'symptom_status, 'race', 'current_status', ‘'state_fips_code', 'exposure_yn'

df7=df.drop(["case_month", "state_fips_code", "exposure_yn","case_onset_interval", "case_positive_specimen_interval","county_fips_code"], axis=1) #columns to definitely drop
print("Before: ", df7.shape)
df7.dropna(inplace=True)
df7bool=~df7.isin(["Unknown", "Missing"])#returns bool dataframe (false if unknown or missing)
df7b=df7bool.all(axis=1)#returns bool series (True if the row has all true values)
df7after=df7.loc[df7b]
print("After: ", df7after.shape)
df7after.to_csv("CovData/df7.csv", index=False)
#'icu_yn', 'hosp_yn', 'age_group,'county_fips_code', 'process',  'underlying_conditions_yn', 'ethnicity,  'sex', 'symptom_status, 'race', 'current_status', ‘'state_fips_code', 'exposure_yn'
#
df8=df.drop(["case_month","exposure_yn","case_onset_interval", "case_positive_specimen_interval"], axis=1) #columns to definitely drop
print("Before: ", df8.shape)
df8.dropna(inplace=True)
df8bool=~df8.isin(["Unknown", "Missing"])#returns bool dataframe (false if unknown or missing)
df8b=df8bool.all(axis=1)#returns bool series (True if the row has all true values)
df8after=df8.loc[df8b]
print("After: ", df8after.shape)
df8after.to_csv("CovData/df8.csv", index=False)

# df7=df.drop(["case_month", "ethnicity", "case_positive_specimen_interval", "case_onset_interval", "state_fips_code"], axis=1) #columns to definitely drop
# print("Before: ", df7.shape)
# df7.dropna(inplace=True)
# df7bool=~df7.isin(["Unknown", "Missing"])#returns bool dataframe (false if unknown or missing)
# df7b=df7bool.all(axis=1)#returns bool series (True if the row has all true values)
# df7after=df7.loc[df7b]
# print("After: ", df7after.shape)
# df7after.to_csv("CovData/df7.csv", index=False)

#"case_month", "ethnicity", "case_positive_specimen_interval", "case_onset_interval", "state_fips_code"
df9=df.drop(["case_month","current_status", "state_fips_code", "exposure_yn","case_onset_interval", "case_positive_specimen_interval","county_fips_code"], axis=1) #columns to definitely drop
print("Before: ", df9.shape)
df9.dropna(inplace=True)
df9bool=~df9.isin(["Unknown", "Missing"])#returns bool dataframe (false if unknown or missing)
df9b=df9bool.all(axis=1)#returns bool series (True if the row has all true values)
df9after=df9.loc[df9b]
print("After: ", df9after.shape)
df9after.to_csv("CovData/df9.csv", index=False)
####this df is for the tree. It does its own feature selection, however I am deleting columns with large NA counts








df10=df.drop(["case_month","case_onset_interval", "case_positive_specimen_interval","county_fips_code"], axis=1) #columns to definitely drop
print("Before: ", df10.shape)
df10.dropna(inplace=True)
df10bool=~df10.isin(["Unknown", "Missing"])#returns bool dataframe (false if unknown or missing)
df10b=df10bool.all(axis=1)#returns bool series (True if the row has all true values)
df10after=df10.loc[df10b]
print("After: ", df10after.shape)
df10after.to_csv("CovData/df10.csv", index=False)

dftree1=df.drop(["case_month","case_onset_interval", "case_positive_specimen_interval", "underlying_conditions_yn","county_fips_code"], axis=1) #columns to definitely drop
print("Before: ", dftree1.shape)
dftree1.dropna(inplace=True)
dftree1bool=~dftree1.isin(["Unknown", "Missing"])#returns bool dataframe (false if unknown or missing)
dftree1b=dftree1bool.all(axis=1)#returns bool series (True if the row has all true values)
dftree1after=dftree1.loc[dftree1b]
print("After: ", dftree1after.shape)
dftree1after.to_csv("CovData/dftree1.csv", index=False)

dftree2=df.drop(["case_month","case_onset_interval", "case_positive_specimen_interval", "ethnicity", "underlying_conditions_yn","county_fips_code"], axis=1) #columns to definitely drop
print("Before: ", dftree2.shape)
dftree2.dropna(inplace=True)
dftree2bool=~dftree2.isin(["Unknown", "Missing"])#returns bool dataframe (false if unknown or missing)
dftree2b=dftree2bool.all(axis=1)#returns bool series (True if the row has all true values)
dftree2after=dftree2.loc[dftree2b]
print("After: ", dftree2after.shape)
dftree2after.to_csv("CovData/dftree2.csv", index=False)

dftree3=df.drop(["case_month","case_onset_interval", "case_positive_specimen_interval", "ethnicity", "underlying_conditions_yn", "exposure_yn","county_fips_code"], axis=1) #columns to definitely drop
print("Before: ", dftree3.shape)
dftree3.dropna(inplace=True)
dftree3bool=~dftree3.isin(["Unknown", "Missing"])#returns bool dataframe (false if unknown or missing)
dftree3b=dftree3bool.all(axis=1)#returns bool series (True if the row has all true values)
dftree3after=dftree3.loc[dftree3b]
print("After: ", dftree3after.shape)
dftree3after.to_csv("CovData/dftree3.csv", index=False)

dftree4=df.drop(["case_month","case_onset_interval", "case_positive_specimen_interval", "process", "underlying_conditions_yn", "exposure_yn","county_fips_code"], axis=1) #columns to definitely drop
print("Before: ", dftree4.shape)
dftree4.dropna(inplace=True)
dftree4bool=~dftree4.isin(["Unknown", "Missing"])#returns bool dataframe (false if unknown or missing)
dftree4b=dftree4bool.all(axis=1)#returns bool series (True if the row has all true values)
dftree4after=dftree4.loc[dftree4b]
print("After: ", dftree4after.shape)
dftree4after.to_csv("CovData/dftree4.csv", index=False)

dftree5=df.drop(["case_month","case_onset_interval", "case_positive_specimen_interval", "process", "underlying_conditions_yn", "exposure_yn", "ethnicity","county_fips_code"], axis=1) #columns to definitely drop
print("Before: ", dftree5.shape)
dftree5.dropna(inplace=True)
dftree5bool=~dftree5.isin(["Unknown", "Missing"])#returns bool dataframe (false if unknown or missing)
dftree5b=dftree5bool.all(axis=1)#returns bool series (True if the row has all true values)
dftree5after=dftree5.loc[dftree5b]
print("After: ", dftree5after.shape)
dftree5after.to_csv("CovData/dftree5.csv", index=False)