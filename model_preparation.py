import pandas as pd
import numpy as np

# Parashikimi nëse një person do të arrestohet bazuar në të dhënat e tij

df = pd.read_csv('police_project.csv')

# print("Informatat per datasetin para preprocesimit")
print(df.info())
# print("Numri i rreshtave: ", df.shape[0])
# print("Numri i kolonave: ", df.shape[1])

# print(df.isna().sum())

# E fshijme kolonen county_name nga dataseti pasi qe te gjitha vlerat e kesaj kolone jane missing
df = df.drop(['county_name'], axis=1)
# Kolona search_type i ka 97% te vlerave missing por qe te dhenat e saj mund te na hyjn ne pune gjate analizes
# andaj vlerat e saj missing i bejme fill me Unknown dhe vlerat aktuale nuk i ndryshojm
df['search_type'] = df['search_type'].fillna('Unknown')
# Rreshtat tek te cilat mungojne vlerat ne te gjitha keto kolona duhet te fshihen sepse nuk permbajn informacion te mjaftueshem qe te bejme prediction
df = df.dropna(subset=['driver_gender', 'driver_age_raw', 'driver_race', 'violation_raw', 'violation', 'stop_outcome', 'is_arrested', 'stop_duration'])

df['stop_datetime'] = pd.to_datetime(df['stop_date'] + ' ' + df['stop_time'])
df = df.drop(['stop_date', 'stop_time'], axis=1)

# Llogaritja e moshes dhe vendosja e saj per personat qe ju mungon
df['driver_age'] = df['driver_age'].fillna(df['driver_age_raw'] - df['stop_datetime'].dt.year)

# print("Tipet e te dhenave")
# print(df.dtypes)

print("Te dhenat pas preprocesimit")
print(df.info())
# print(df.isna().sum())

df.to_csv('preprocessed_poloce_project.csv', index=False)