import pandas as pd
import numpy as np
df = pd.read_csv("dulieuxettuyendaihoc.csv")


#Cau6
diem_cols = [col for col in df.columns 
             if col.startswith(('T','L','H','S','V','X','D','N','DH'))]
for col in diem_cols:
    df[col] = df[col].fillna(df[col].mean())

#Cau7
def tinh_TBM(T, L, H, S, V, X, D, N):
    return (T*2 + L + H + S + V*2 + X + D + N) / 10
df['TBM1'] = tinh_TBM(df['T1'], df['L1'], df['H1'], df['S1'],
                      df['V1'], df['X1'], df['D1'], df['N1'])
df['TBM2'] = tinh_TBM(df['T2'], df['L2'], df['H2'], df['S2'],
                      df['V2'], df['X2'], df['D2'], df['N2'])
df['TBM3'] = tinh_TBM(df['T6'], df['L6'], df['H6'], df['S6'],
                      df['V6'], df['X6'], df['D6'], df['N6'])

#Cau8
def xep_loai(tbm):
    if tbm < 5:
        return 'Y'
    elif tbm < 6.5:
        return 'TB'
    elif tbm < 8:
        return 'K'
    elif tbm < 9:
        return 'G'
    else:
        return 'XS'
df['XL1'] = df['TBM1'].apply(xep_loai)
df['XL2'] = df['TBM2'].apply(xep_loai)
df['XL3'] = df['TBM3'].apply(xep_loai)

#Cau9
def minmax_scale_4(x):
    return (x - x.min()) / (x.max() - x.min()) * 4
df['US_TBM1'] = minmax_scale_4(df['TBM1'])
df['US_TBM2'] = minmax_scale_4(df['TBM2'])
df['US_TBM3'] = minmax_scale_4(df['TBM3'])

#Cau10
def xet_tuyen(row):
    if row['KT'] in ['A', 'A1']:
        diem = (row['DH1']*2 + row['DH2'] + row['DH3']) / 4
    elif row['KT'] == 'B':
        diem = (row['DH1'] + row['DH2']*2 + row['DH3']) / 4
    else:
        diem = (row['DH1'] + row['DH2'] + row['DH3']) / 3
    return 1 if diem >= 5 else 0
df['KQXT'] = df.apply(xet_tuyen, axis=1)

#Cau11
df.to_csv("processed_dulieuxettuyendaihoc.csv", index=False)
