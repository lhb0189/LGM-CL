import pandas as pd
df=pd.read_csv('muv_test.csv')
smiles=df["smiles"].tolist()
print(len(smiles))
"""
drop_list=[]
for i in range(0,186174,2):
    drop_list.append(i)
df=df.drop(df.index[drop_list]).reset_index(drop=True)
df.to_csv("muv_test.csv",index=False)
"""