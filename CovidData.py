import pandas as pd
import matplotlib.pyplot as plt
#df=pd.read_csv("COVID-19_Case_Surveillance_Public_Use_Data_with_Geography.csv", parse_dates=["case_month"], infer_datetime_format=True, usecols=["case_month", "state_fips_code", "county_fips_code", "age_group", "sex", "race", "ethnicity", "case_positive_specimen_interval", "case_onset_interval", "process", "exposure_yn", "current_status", "symptom_status", "hosp_yn", "icu_yn", "death_yn", "underlying_conditions_yn"], dtype={"current_Status":str})
df=pd.read_csv("COVID-19_Case_Surveillance_Public_Use_Data_with_Geography.csv",parse_dates=["case_month"], infer_datetime_format=True, usecols=["case_month", "state_fips_code", "county_fips_code", "age_group", "sex", "race", "ethnicity", "case_positive_specimen_interval", "case_onset_interval", "process", "exposure_yn", "current_status", "symptom_status", "hosp_yn", "icu_yn", "death_yn", "underlying_conditions_yn"], dtype={"state_fips_code":"Int64", "county_fips_code":"Int64", "case_positive_specimen_interval":"Int64", "case_onset_interval":"Int64"})
#print(df.shape)
df2=df.drop("case_month", axis=1).dropna()
for col in df2.columns:
    val=df2[col].value_counts()
    val["NA"]=df[col].isna().sum() #need to add column for NA manually
    gTitle=col+" Value Frequencies"
    val.plot(kind="bar", rot=15, title=gTitle, ylabel="Frequency", fontsize=5, figsize=(7.5, 5.5))
    fname="Column_counts/"+col+".png"
    plt.savefig(fname)


# dfDropNA=df.dropna()
# dfDropNA.drop(["Unknown", "Missing"], inplace=True)
# print(dfDropNA.shape)
# df.to_csv("edit.csv")
# dfDropNA("NoNA.csv")