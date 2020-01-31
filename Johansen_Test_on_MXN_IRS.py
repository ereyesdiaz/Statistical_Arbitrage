# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 15:35:41 2018
@author: Enrique Reyes
Descrption.- This code apply Johansen test to check for Co-integration on FX pairs
and the establish a Mean Reversal trading strategy using Bollinger Bands logic. 
"""

import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from johansen import coint_johansen

##################################### Data ##################################################
data=pd.read_csv("./MXN_IRS.csv") # Get Market Historical Data
columns=['Date','RateX','RateY'] # Define Initial Columns for Strategy DataFrame
df=pd.DataFrame(index=data.index,columns=columns) # Create Strategy DataFrame
df['Date']=data['Date'] # Filling Date Column
df['RateX']=data['10Y'] # Filling Short_Rate Column - Will change depending Spread
df['RateY']=data['20Y'] # Filling Long_Rate Column - Will change depending Spread
############################### Define Strategy Parameters ##################################
Initial_Capital=1000000   # USD Working Capital
DV01=10000                # 10,000 USD x bps Constat Risk Sensitivity for every trade
Slippage=0.0             # Slippage Cost in bps
Commission=0.00           # Commission in bps
ADF_sl=0.95               # Augmented Dickey Fuller Test Significance Level
tau=1/252                 # Augmented Dickey Fuller Test Significance Level
############################## Parameters to Optimize #######################################
Z=0.7
Stop_mult=10      # Stop Loss will be calculated in multiples of Rolling StdDev
test=60           # Rolling Window Size for Co-integration tests
trade=10
############################## Engle-Granger procedure loop #################################
df['B1_const']=np.zeros(len(df))  
df['B2_RateX']=np.zeros(len(df))  
df['OLS_pvalue']=np.zeros(len(df))  
df['ADF_p-value']=np.zeros(len(df))  
df['ADF_Stationarity']=np.zeros(len(df))  
df['ErrCor_Beta']=np.zeros(len(df))  
df['ErrCor_p-value']=np.zeros(len(df))  
df['Stationarity']=np.zeros(len(df))  
df['theta']=np.zeros(len(df))  
df['Half_Life']=np.zeros(len(df))  
df['MiuE']=np.zeros(len(df))  
df['SigmaEQ']=np.zeros(len(df))  
df['Upper_Bd']=np.zeros(len(df))  
df['Lower_Bd']=np.zeros(len(df))  
N = len(df)
n = int((N-test)/trade)
for i in range(0,n+1):
    it = i*trade
    iT = test+i*trade
    if ((test+trade+i*trade)>N):
        iN = N
    else:
        iN = (test+trade+i*trade)
    X1=df['RateX'][it:iT]
    Y1=df['RateY'][it:iT]
    X1=sm.add_constant(X1)
    lm1=sm.OLS(Y1,X1).fit()
    b1lm1=lm1.params['const'] 
    b2lm1=lm1.params['RateX']
    pvalue = lm1.pvalues[1]
    df.at[df.index[iT:iN],'B1_const']= b1lm1
    df.at[df.index[iT:iN],'B2_RateX']= b2lm1
    df.at[df.index[iT:iN],'OLS_pvalue']= pvalue
    X1=df['RateX'][it:iT]
    Et=Y1-(b2lm1*X1)-b1lm1
    ADFpvalue=adfuller(Et)[1]
    Et1 = Et.shift(1)
    if (pvalue<0.05):
        df.at[df.index[iT:iN],'ADF_p-value']= ADFpvalue
        df.at[df.index[iT:iN],'ADF_Stationarity']= 1
        Xd = X1.diff()
        Yd = Y1.diff()
        x1 = Xd.iloc[1:]
        Y2 = Yd.iloc[1:]
        x2 = Et1.iloc[1:]
        x1.name = 'RateXd'
        x2.name = 'Et-1'
        X2 = pd.concat([x1,x2],axis=1)
        lm2=sm.OLS(Y2,X2).fit()
        b2lm2=lm2.params['Et-1']
        pvalue2 = lm2.pvalues[1]
        df.at[df.index[iT:iN],'ErrCor_Beta']= b2lm2
        df.at[df.index[iT:iN],'ErrCor_p-value']= pvalue
        if (pvalue2<0.05):
            df.at[df.index[iT:iN],'Stationarity']= 1
    et=Y1-b2lm1*X1
    et1=et.shift(1)
    et1=et1[1:]
    et=et[1:]
    et.name = 'et'
    et1.name = 'et-1'
    X3=sm.add_constant(et1)
    Y3=et
    lm3=sm.OLS(Y3,X3).fit()
    C=lm3.params['const'] 
    B=lm3.params['et-1']
    theta=-np.log(B)/tau
    MiuE=C/(1-B)
    SSE=lm3.ssr
    n=lm3.nobs
    STD = np.sqrt(SSE/(n-1))
    yrfrac=np.log(2)/theta   
    HL_wdays=yrfrac/tau
    SigmaOU = STD*np.sqrt(-2*np.log(B)/(tau*(1-(B**2))))
    SigmaEQ = SigmaOU/np.sqrt(2*theta)
    df.at[df.index[iT:iN],'theta']= theta
    df.at[df.index[iT:iN],'Half_Life']= HL_wdays
    df.at[df.index[iT:iN],'MiuE']= MiuE
    df.at[df.index[iT:iN],'SigmaEQ']= SigmaEQ
    df.at[df.index[iT:iN],'Upper_Bd']= MiuE+(Z*SigmaEQ)
    df.at[df.index[iT:iN],'Lower_Bd']= MiuE-(Z*SigmaEQ)  

df = df[test:]
df.index = range(0,N-test)
df['Et']=df['RateY']-(df['B2_RateX']*df['RateX'])
df.at[df.index[0],'Et']= 0
##############################  Define Trade Management Variables ##############################
df['Signal'] = np.zeros(len(df))             # If Signal=1 Pay/Rec Long/Short_Rate If=-1 Rec/Pay
df['MktPos'] = np.zeros(len(df))             # Will be 1 for Longs, 0 Flat, -1 for Shorts
df['Entry_RateY'] = np.zeros(len(df))        # Will save Entry Price when occurs for Longs or Shorts 
df['Entry_RateX'] = np.zeros(len(df))        # Will save Entry Price when occurs for Longs or Shorts
df['Exit_RateY'] = np.zeros(len(df))         # Will save Exit Price when occurs for Longs or Shorts
df['Exit_RateX'] = np.zeros(len(df))         # Will save Exit Price when occurs for Longs or Shorts
df['DV01_RateY'] = np.zeros(len(df))         # Will save Entry Price when occurs for Longs or Shorts 
df['DV01_RateX'] = np.zeros(len(df))         # Will save Entry Price when occurs for Longs or Shorts
df['EntryRule'] = ''                         # Will save a string describing the Entry Rule
df['ExitRule'] = ''                          # Will save a string describing the Exit Rule
df['Slippage'] = np.zeros(len(df))           # Compute Slippage P&L 
df['Commission'] = np.zeros(len(df))         # Compute Commission P&L
df['Daily_PnL'] = np.zeros(len(df))          # Compute Daily P&L 
df['Cum_PnL'] = np.zeros(len(df))            # Total PnL per Trade
df['Longs'] = np.zeros(len(df))              # Will compute Long trades at the end
df['Shorts'] = np.zeros(len(df))             # Will compute Short trade at the end
df['Trade_PnL'] = np.zeros(len(df))          # Total PnL per Trade
df['Daily_Rtn'] = np.zeros(len(df))          # Compute Ln Returns for each bar used 
df['Trade_Rtn'] = np.zeros(len(df))          # Simple Return at the end of each trade 
df['Equity'] = np.ones(len(df))*Initial_Capital
################################ Running loop to generate trades ################################
N = len(df)
for i in range(1,N):
    ################################################ If Market Position = 0 #####################
    if (df['MktPos'][i-1]==0):
        ##################################### Entry Rule for Long Trades ########################
        if ((df['Stationarity'][i]>0)and(df['Et'][i]<df['Lower_Bd'][i])):
            entrybar = i
            entryRateY = df['RateY'][i]
            entryRateX = df['RateX'][i]
            DV01_RateY = DV01 
            DV01_RateX = -DV01*df['B2_RateX'][i]
            df.at[df.index[i],'Signal']= 1
            df.at[df.index[i],'Entry_RateY']= entryRateY
            df.at[df.index[i],'Entry_RateX']= entryRateX
            df.at[df.index[i],'DV01_RateY']= DV01_RateY
            df.at[df.index[i],'DV01_RateX']= DV01_RateX
            df.at[df.index[i],'MktPos']= 1
            df.at[df.index[i],'EntryRule']= 'Long_Entry'
            df.at[df.index[i],'Slippage']= (-DV01_RateY+DV01_RateX)*Slippage
            df.at[df.index[i],'Commission']= (-DV01_RateY+DV01_RateX)*Commission
            df.at[df.index[i],'Daily_PnL']= (df['Slippage'][i]+df['Commission'][i])
            InvestedEq = df['Equity'][i-1]
            df.at[df.index[i],'Equity']= (InvestedEq + df['Daily_PnL'][i])
            df.at[df.index[i],'Daily_Rtn']= round((np.log(df['Equity'][i]/df['Equity'][i-1]))*100,2)
        ##################################### Entry Rule for Short Trades #######################
        elif ((df['Stationarity'][i]>0)and(df['Et'][i]>df['Upper_Bd'][i])): 
            entrybar = i
            entryRateY = df['RateY'][i]
            entryRateX = df['RateX'][i]
            DV01_RateY = -DV01 
            DV01_RateX = DV01*df['B2_RateX'][i]
            df.at[df.index[i],'Signal']= -1
            df.at[df.index[i],'Entry_RateY']= entryRateY
            df.at[df.index[i],'Entry_RateX']= entryRateX
            df.at[df.index[i],'DV01_RateY']= DV01_RateY
            df.at[df.index[i],'DV01_RateX']= DV01_RateX
            df.at[df.index[i],'MktPos']= -1
            df.at[df.index[i],'EntryRule']= 'Short_Entry'
            df.at[df.index[i],'Slippage']= (DV01_RateY-DV01_RateX)*Slippage
            df.at[df.index[i],'Commission']= (DV01_RateY-DV01_RateX)*Commission
            df.at[df.index[i],'Daily_PnL']= (df['Slippage'][i]+df['Commission'][i])
            InvestedEq = df['Equity'][i-1]
            df.at[df.index[i],'Equity']= (InvestedEq + df['Daily_PnL'][i])
            df.at[df.index[i],'Daily_Rtn']= round((np.log(df['Equity'][i]/df['Equity'][i-1]))*100,2)
        else:
            df.at[df.index[i],'Equity']= df['Equity'][i-1]  # If there is no entry then Equity remains
    ####################################### If we are in a Long Trade then #######################
    elif (df['MktPos'][i-1]>0):
        ###################################### Check for Stop Loss ###############################
        if (df['Cum_PnL'][i-1]<(-Stop_mult*DV01)):
            df.at[df.index[i],'Signal']= -1
            df.at[df.index[i],'ExitRule']= 'Stop_Long'
            df.at[df.index[i],'Longs']= 1                     
            df.at[df.index[i],'MktPos']= 0
            exitRateY = df['RateY'][i]
            exitRateX = df['RateX'][i]
            df.at[df.index[i],'Exit_RateY']= exitRateY
            df.at[df.index[i],'Exit_RateX']= exitRateX
            df.at[df.index[i],'Slippage']= (-DV01_RateY+DV01_RateX)*Slippage
            df.at[df.index[i],'Commission']= (-DV01_RateY+DV01_RateX)*Commission
            df.at[df.index[i],'Daily_PnL']= round(((df['RateY'][i]-df['RateY'][i-1])*100*DV01_RateY)+((df['RateX'][i]-df['RateX'][i-1])*100*DV01_RateX)+(df['Slippage'][i]+df['Commission'][i]),0)
            df.at[df.index[i],'Cum_PnL']= round(df['Daily_PnL'][entrybar:(i+1)].sum(),0)
            df.at[df.index[i],'Equity']= (df['Equity'][i-1] + df['Daily_PnL'][i])
            df.at[df.index[i],'Daily_Rtn']= round((np.log(df['Equity'][i]/df['Equity'][i-1]))*100,2)
            df.at[df.index[i],'Trade_Rtn']= round(((exitRateY-entryRateY)*DV01_RateY*100+(exitRateX-entryRateX)*DV01_RateX*100)*100/InvestedEq,2)
            df.at[df.index[i],'Trade_PnL']= ((exitRateY-entryRateY)*DV01_RateY*100)+((exitRateX-entryRateX)*DV01_RateX*100)+((-DV01_RateY+DV01_RateX)*(Slippage+Commission))
        ################################### Check for Profit Taking Exit #########################
        elif (df['Et'][i]>df['MiuE'][i]):
            df.at[df.index[i],'Signal']= -1
            df.at[df.index[i],'ExitRule']= 'Exit_Long'
            df.at[df.index[i],'Longs']= 1                               
            df.at[df.index[i],'MktPos']= 0
            exitRateY = df['RateY'][i]
            exitRateX = df['RateX'][i]
            df.at[df.index[i],'Exit_RateY']= exitRateY
            df.at[df.index[i],'Exit_RateX']= exitRateX
            df.at[df.index[i],'Slippage']= (-DV01_RateY+DV01_RateX)*Slippage
            df.at[df.index[i],'Commission']= (-DV01_RateY+DV01_RateX)*Commission
            df.at[df.index[i],'Daily_PnL']= round(((df['RateY'][i]-df['RateY'][i-1])*100*DV01_RateY)+((df['RateX'][i]-df['RateX'][i-1])*100*DV01_RateX)+(df['Slippage'][i]+df['Commission'][i]),0)
            df.at[df.index[i],'Cum_PnL']= round(df['Daily_PnL'][entrybar:(i+1)].sum(),0)
            df.at[df.index[i],'Equity']= (df['Equity'][i-1] + df['Daily_PnL'][i])
            df.at[df.index[i],'Daily_Rtn']= round((np.log(df['Equity'][i]/df['Equity'][i-1]))*100,2)
            df.at[df.index[i],'Trade_Rtn']= round(((exitRateY-entryRateY)*DV01_RateY*100+(exitRateX-entryRateX)*DV01_RateX*100)*100/InvestedEq,2)
            df.at[df.index[i],'Trade_PnL']= ((exitRateY-entryRateY)*DV01_RateY*100)+((exitRateX-entryRateX)*DV01_RateX*100)+((-DV01_RateY+DV01_RateX)*(Slippage+Commission))
        #################################### If no Exit condition applies ########################
        else:
            df.at[df.index[i],'MktPos']= 1
            df.at[df.index[i],'DV01_RateY']= DV01_RateY
            df.at[df.index[i],'DV01_RateX']= DV01_RateX
            df.at[df.index[i],'Daily_PnL']= round(((df['RateY'][i]-df['RateY'][i-1])*100*DV01_RateY)+((df['RateX'][i]-df['RateX'][i-1])*100*DV01_RateX),0)
            df.at[df.index[i],'Cum_PnL']= round(df['Daily_PnL'][entrybar:(i+1)].sum(),0)
            df.at[df.index[i],'Equity']= (df['Equity'][i-1] + df['Daily_PnL'][i])
            df.at[df.index[i],'Daily_Rtn']= round((np.log(df['Equity'][i]/df['Equity'][i-1]))*100,2)
    ############################################### If we are in a Short Trade ###################
    elif (df['MktPos'][i-1]<0):
        if (df['Cum_PnL'][i-1]<(-Stop_mult*DV01)):
            df.at[df.index[i],'Signal']= 1
            df.at[df.index[i],'ExitRule']= 'Stop_Short'
            df.at[df.index[i],'Shorts']= 1                               
            df.at[df.index[i],'MktPos']= 0
            exitRateY = df['RateY'][i]
            exitRateX = df['RateX'][i]
            df.at[df.index[i],'Exit_RateY']= exitRateY
            df.at[df.index[i],'Exit_RateX']= exitRateX
            df.at[df.index[i],'Slippage']= (DV01_RateY-DV01_RateX)*Slippage
            df.at[df.index[i],'Commission']= (DV01_RateY-DV01_RateX)*Commission
            df.at[df.index[i],'Daily_PnL']= round(((df['RateY'][i]-df['RateY'][i-1])*100*DV01_RateY)+((df['RateX'][i]-df['RateX'][i-1])*100*DV01_RateX)+(df['Slippage'][i]+df['Commission'][i]),0)
            df.at[df.index[i],'Cum_PnL']= round(df['Daily_PnL'][entrybar:(i+1)].sum(),0)
            df.at[df.index[i],'Equity']= (df['Equity'][i-1] + df['Daily_PnL'][i])
            df.at[df.index[i],'Daily_Rtn']= round((np.log(df['Equity'][i]/df['Equity'][i-1]))*100,2)
            df.at[df.index[i],'Trade_Rtn']= round(((exitRateY-entryRateY)*DV01_RateY*100+(exitRateX-entryRateX)*DV01_RateX*100)*100/InvestedEq,2)
            df.at[df.index[i],'Trade_PnL']= ((exitRateY-entryRateY)*DV01_RateY*100)+((exitRateX-entryRateX)*DV01_RateX*100)+((DV01_RateY-DV01_RateX)*(Slippage+Commission))
        elif (df['Et'][i]<df['MiuE'][i]):
            df.at[df.index[i],'Signal']= 1
            df.at[df.index[i],'ExitRule']= 'Exit_Short'
            df.at[df.index[i],'Shorts']= 1                               
            df.at[df.index[i],'MktPos']= 0
            exitRateY = df['RateY'][i]
            exitRateX = df['RateX'][i]
            df.at[df.index[i],'Exit_RateY']= exitRateY
            df.at[df.index[i],'Exit_RateX']= exitRateX
            df.at[df.index[i],'Slippage']= (DV01_RateY-DV01_RateX)*Slippage
            df.at[df.index[i],'Commission']= (DV01_RateY-DV01_RateX)*Commission
            df.at[df.index[i],'Daily_PnL']= round(((df['RateY'][i]-df['RateY'][i-1])*100*DV01_RateY)+((df['RateX'][i]-df['RateX'][i-1])*100*DV01_RateX)+(df['Slippage'][i]+df['Commission'][i]),0)
            df.at[df.index[i],'Cum_PnL']= round(df['Daily_PnL'][entrybar:(i+1)].sum(),0)
            df.at[df.index[i],'Equity']= (df['Equity'][i-1] + df['Daily_PnL'][i])
            df.at[df.index[i],'Daily_Rtn']= round((np.log(df['Equity'][i]/df['Equity'][i-1]))*100,2)
            df.at[df.index[i],'Trade_Rtn']= round(((exitRateY-entryRateY)*DV01_RateY*100+(exitRateX-entryRateX)*DV01_RateX*100)*100/InvestedEq,2)
            df.at[df.index[i],'Trade_PnL']= ((exitRateY-entryRateY)*DV01_RateY*100)+((exitRateX-entryRateX)*DV01_RateX*100)+((DV01_RateY-DV01_RateX)*(Slippage+Commission))
        else:
            df.at[df.index[i],'MktPos']= -1
            df.at[df.index[i],'DV01_RateY']= DV01_RateY
            df.at[df.index[i],'DV01_RateX']= DV01_RateX
            df.at[df.index[i],'Daily_PnL']= round(((df['RateY'][i]-df['RateY'][i-1])*100*DV01_RateY)+((df['RateX'][i]-df['RateX'][i-1])*100*DV01_RateX),0)
            df.at[df.index[i],'Cum_PnL']= round(df['Daily_PnL'][entrybar:(i+1)].sum(),0)
            df.at[df.index[i],'Equity']= (df['Equity'][i-1] + df['Daily_PnL'][i])
            df.at[df.index[i],'Daily_Rtn']= round((np.log(df['Equity'][i]/df['Equity'][i-1]))*100,2)

#################################### Calculate Output Variables ###################################
Longs_PnL = df['Trade_PnL']*df['Longs']
Shorts_PnL = df['Trade_PnL']*df['Shorts']
Long_Winners = float(sum(float(num)>0 for num in Longs_PnL))
Short_Winners = float(sum(float(num)>0 for num in Shorts_PnL))
Total_Longs = df['Longs'].sum()
Total_Shorts = df['Shorts'].sum()
Long_Losers = float(sum(float(num)<0 for num in Longs_PnL))
Short_Losers = float(sum(float(num)<0 for num in Shorts_PnL))
Hit_R_Longs = round((Long_Winners/Total_Longs)*100,2)
Hit_R_Shorts = round((Short_Winners/Total_Shorts)*100,2)
Winners_PnL_L = round(Longs_PnL[Longs_PnL>0].sum(),2)
Losers_PnL_L = round(Longs_PnL[Longs_PnL<0].sum(),2)
Winners_PnL_S = round(Shorts_PnL[Shorts_PnL>0].sum(),2)
Losers_PnL_S = round(Shorts_PnL[Shorts_PnL<0].sum(),2)
if (Long_Winners == 0):
    Avg_Profit_L = 0
else:
    Avg_Profit_L = round(Winners_PnL_L/Long_Winners,4)
if (Long_Losers == 0):
    Avg_Loss_L = 0
else:
    Avg_Loss_L = round(Losers_PnL_L/Long_Losers,4)
#Avg_PL_R_L = round(Avg_Profit_L/(-Avg_Loss_L),4)

if (Short_Winners == 0):
    Avg_Profit_S = 0
else:
    Avg_Profit_S = round(Winners_PnL_S/Short_Winners,4)
if (Short_Losers == 0):
    Avg_Loss_S = 0
else:
    Avg_Loss_S = round(Losers_PnL_S/Short_Losers,4)
#Avg_PL_R_S = round(Avg_Profit_S/(-Avg_Loss_S),4)
df['MaxEquity'] = np.ones(len(df))*Initial_Capital      
for j in range(0,N):
    df.at[df.index[j],'MaxEquity']= df['Equity'][0:(j+1)].max()
df['DD'] = round((df['Equity']/df['MaxEquity']-1)*100,2)
MaxDD = df['DD'].min()
tempcolnum = df.columns.get_loc('Equity')
Total_Return = round(((df.iloc[N-1, tempcolnum]/Initial_Capital)-1)*100,2)
Total_Trades = float(sum(float(num)!=0 for num in df['Trade_Rtn']))
PctLongs = Total_Longs/Total_Trades
Winners = float(sum(float(num)>0 for num in df['Trade_Rtn']))
Losers = float(sum(float(num)<0 for num in df['Trade_Rtn']))
Years = 2.5
Trades_xYr = round(Total_Trades / Years,1)
if (Total_Trades == 0):
    Hit_Ratio = 0
else:
    Hit_Ratio = round((Winners/Total_Trades)*100,2)
Winners_PnL = round(df[df.Trade_PnL>0].sum()['Trade_PnL'],2)
Losers_PnL = round(df[df.Trade_PnL<0].sum()['Trade_PnL'],2)
if (Winners == 0):
    Avg_Profit = 0
else:
    Avg_Profit = round(Winners_PnL/Winners,2)
if (Losers == 0):
    Avg_Loss = 0
else:
    Avg_Loss = round(Losers_PnL/Losers,2)
Avg_PL_Ratio = round(Avg_Profit/(-Avg_Loss),2)
Daily_AvgLnRtn = np.mean(df['Daily_Rtn'])
Daily_Std = np.std(df['Daily_Rtn'])
if (Daily_Std == 0):
    SharpeRatio = 0
else:
    SharpeRatio = round((Daily_AvgLnRtn/Daily_Std)*np.sqrt(252),2)
Cagr = round((((1+(Total_Return/100))**(float(1)/Years))-1)*100,2) 
df = df.set_index(['Date'])
Returns = df['Daily_Rtn']
Equity = df['Equity']
DD_curve = df['DD']   
############################# Plot Equity and Drawdown Curve #####################
Equity.plot(figsize=(9,5))
plt.title('Equity Curve')
plt.show()
DD_curve.plot(figsize=(9,3), color='r', kind='area')
plt.title('Draw Down Curve')
plt.show()


print('Strategy Performance Summary')
print('Total Return      ', Total_Return)
print('CAGR              ', Cagr)
print('Sharpe Ratio      ', SharpeRatio)
print('Hit Ratio         ', Hit_Ratio)
print('Avg. P/L Ratio    ', Avg_PL_Ratio)
print('MaxDD             ', MaxDD)
print('Total Trades      ', Total_Trades)
print('Trades x Yr       ', Trades_xYr)
print('Longs Trades      ', Total_Longs)
print('Short Trades      ', Total_Shorts)
