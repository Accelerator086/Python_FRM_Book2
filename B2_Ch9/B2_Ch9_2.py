# B2_Ch9_2.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

import pandas as pd
import matplotlib.pyplot as plt

CDS_spreads = pd.read_csv("C:\\Dropbox\\FRM Book\\CreditRisk\\CDS_spreads.csv")
df = pd.DataFrame(CDS_spreads)

df['Survival'] = 0.0
numerator1 = 0.0
numerator2 = 0.0
denominator1 = 0.0
denominator2 = 0.0
term_final = 0.0

df['Spread'] = df['Spread']/10000

RR = df.at[0,'Recovery']
L = 1.0 - RR

t1 = df.at[0, 'Maturity']
t2 = df.at[1, 'Maturity']
delta_t = t2-t1

for row_index,row in df.iterrows():
    if(row_index == 0):
        df.at[0,'Survival'] = 1
    if(row_index==1):
        df.at[1,'Survival'] = L / (L + (delta_t * df.at[1,'Spread']))
    if(row_index>1):
        temp_counter = row_index
        term1 = 0.0
        term2 = 0.0
        j = 1
        while(j > row_index-1):
            numerator1_temp = df.at[row_index,'DF'] * ((L * df.at[row_index - 1,'Survival']) - ((L + (delta_t * df.at[row_index,'Spread']))*(df.at[row_index,'Survival'])))
            numerator1 = numerator1 + numerator1_temp
            row_index = row_index - 1
        row_index = temp_counter
        denominator1 = ((df.at[row_index,'DF']) * (L + (delta_t * df.at[row_index,'Spread'])))
        term1_temp = numerator1/denominator1
        term1 = term1 + term1_temp

        numerator2 = (L * df.at[row_index - 1,'Survival'])
        denominator2 = (L + (delta_t * df.at[row_index,'Spread']))
        term2_temp = numerator2/denominator2
        term2 = term2 + term2_temp
        term_final = term1 + term2
        df.at[row_index, 'Survival'] = term_final
            
plt.plot(df['Maturity'], df['Survival'])
plt.title('Survival probability')
plt.xlabel('Maturity')
plt.ylabel('Survival probability')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')  
