# B2_Ch9_3.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

# calculate Altman Z-Score fore public corporation
def Altman_Z_scorec(WC,TA,RE,EBIT,MVE,TL,S):
    # WC: Working Capital
    # TA: Total Assests
    # RE: Retained Earnings
    # EBIT: Earnings Before Interest and Tax
    # MVE: Market Value of Equity
    # TL: Total Liabilities
    # S: Net Sales
    
    X1 = WC/TA;
    X2 = RE/TA;
    X3 = EBIT/TA;
    X4 = MVE/TL;
    X5 = S/TA;
     
    # calculate z-score
    Z_score = 1.2*X1 + 1.4*X2 + 3.3*X3 + .6*X4 + X5;
    print('Altman value is ', round(Z_score, 2))
    
    # display results
    if Z_score > 3.0:    
        print('Business is healthy.')
    elif Z_score < 1.8:
        print('Business is bankrupt.')
    else: 
        print('Business is intermediate.')
