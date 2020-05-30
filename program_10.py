#!/usr/bin/env python
# coding: utf-8

# In[19]:


#!/bin/env python
# Created on March 25, 2020
#  by Keith Cherkauer
#
# This script serves a as the solution set for assignment-10 on descriptive
# statistics and environmental informatics.  See the assignment documention 
# and repository at:
# https://github.com/Environmental-Informatics/assignment-10.git for more
# details about the assignment.


# In[ ]:


# This program is to process Statistical Data Analysis on the River-flow Data
# Name: Mukhamad Suhermanto
# email: msuherma@purdue.edu


# In[ ]:


import pandas as pd
import scipy.stats as stats
import numpy as np


# In[2]:


def ReadData( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    raw data read from that file in a Pandas DataFrame.  The DataFrame index
    should be the year, month and day of the observation.  DataFrame headers
    should be "agency_cd", "site_no", "Date", "Discharge", "Quality". The 
    "Date" column should be used as the DataFrame index. The pandas read_csv
    function will automatically replace missing values with np.NaN, but needs
    help identifying other flags used by the USGS to indicate no data is 
    availabiel.  Function returns the completed DataFrame, and a dictionary 
    designed to contain all missing value counts that is initialized with
    days missing between the first and last date of the file."""
    
    # define column names
    colNames = ['agency_cd', 'site_no', 'Date', 'Discharge', 'Quality']

    # open and read the file
    DataDF = pd.read_csv(fileName, header=1, names=colNames,  
                         delimiter=r"\s+",parse_dates=[2], comment='#',
                         na_values=['Eqp'])
    DataDF = DataDF.set_index('Date')
    
    #removing negative discharge values as gross error check
    DataDF=DataDF[~(DataDF['Discharge']<0)]
    
    # quantify the number of missing values
    MissingValues = DataDF["Discharge"].isna().sum()
    
    return( DataDF, MissingValues )


# In[3]:


def ClipData( DataDF, startDate, endDate ):
    """This function clips the given time series dataframe to a given range 
    of dates. Function returns the clipped dataframe and and the number of 
    missing values."""
    
    # clipping the data for the range date parameters
    DataDF=DataDF[startDate:endDate]
    
    # quantifying the number of missing value
    MissingValues = DataDF["Discharge"].isna().sum()
      
    return( DataDF, MissingValues )


# In[4]:


def CalcTqmean(Qvalues):
    """This function computes the Tqmean of a series of data, typically
       a 1 year time series of streamflow, after filtering out NoData
       values.  Tqmean is the fraction of time that daily streamflow
       exceeds mean streamflow for each year. Tqmean is based on the
       duration rather than the volume of streamflow. The routine returns
       the Tqmean value for the given data array."""
    
    # dropping the NA value
    Qvalues = Qvalues.dropna()
    
    # computing Tqmean
    Tqmean = ((Qvalues > Qvalues.mean()).sum())/len(Qvalues)
    
    return ( Tqmean )


# In[6]:


def CalcRBindex(Qvalues):
    """This function computes the Richards-Baker Flashiness Index
       (R-B Index) of an array of values, typically a 1 year time
       series of streamflow, after filtering out the NoData values.
       The index is calculated by dividing the sum of the absolute
       values of day-to-day changes in daily discharge volumes
       (pathlength) by total discharge volumes for each year. The
       routine returns the RBindex value for the given data array."""
    
    # dropping the NA value
    Qvalues = Qvalues.dropna()
    
    # summing of abs values
    Tsum = np.abs(Qvalues[:-1].values - Qvalues[1:].values).sum()
    
    # Dividing the sum 
    RBindex = (Tsum / Qvalues[1:].sum())    
        
    return ( RBindex )


# In[7]:


def Calc7Q(Qvalues):
    """This function computes the seven day low flow of an array of 
       values, typically a 1 year time series of streamflow, after 
       filtering out the NoData values. The index is calculated by 
       computing a 7-day moving average for the annual dataset, and 
       picking the lowest average flow in any 7-day period during
       that year.  The routine returns the 7Q (7-day low flow) value
       for the given data array."""
    
    # dropping the NA value
    Qvalues = Qvalues.dropna()
    
    # calculating val7Q
    val7Q = Qvalues.rolling(window=7).mean().min()
    
    return ( val7Q )


# In[8]:


def CalcExceed3TimesMedian(Qvalues):
    """This function computes the number of days with flows greater 
       than 3 times the annual median flow. The index is calculated by 
       computing the median flow from the given dataset (or using the value
       provided) and then counting the number of days with flow greater than 
       3 times that value.   The routine returns the count of events greater 
       than 3 times the median annual flow value for the given data array."""

    # dropping the NA value
    Qvalues = Qvalues.dropna()
    
    # calculating discharge> 3x the annual median
    med3x = (Qvalues>3*Qvalues.median()).sum()
    
    return ( med3x )


# In[20]:


def GetAnnualStatistics(DataDF):
    """This function calculates annual descriptive statistcs and metrics for 
    the given streamflow time series.  Values are retuned as a dataframe of
    annual values for each water year.  Water year, as defined by the USGS,
    starts on October 1."""
    
    # naming columns
    ColNames = ['site_no', 'Mean Flow', 'Peak Flow', 'Median  Flow', 'Coeff Var','Skew','Tqmean','R-B Index','7Q','3xMedian']  
   
    # resampling data of water year
    WYDataDF = DataDF.resample('AS-OCT').mean() 

    # storing annual metric values as a New dataframe
    WYDataDF = pd.DataFrame(0, index=WYDataDF.index,columns=ColNames) 

    WYDataDF['site_no'] =  DataDF['site_no'].mean()
    WYDataDF["Mean Flow"] = DataDF["Discharge"].mean()
    WYDataDF["Peak Flow"] = DataDF["Discharge"].max()
    WYDataDF["Median Flow"] = DataDF["Discharge"].median()
    WYDataDF["Coeff Var"] = DataDF["Discharge"].resample('AS-OCT').std() / WYDataDF["Mean Flow"] * 100.
    WYDataDF["Skew"] = DataDF["Discharge"].resample('AS-OCT').apply(stats.skew)
    WYDataDF["Tqmean"] = DataDF["Discharge"].resample('AS-OCT').apply(CalcTqmean)
    WYDataDF["R-B Index"] = DataDF["Discharge"].resample('AS-OCT').apply(CalcRBindex)
    WYDataDF["7Q"] = DataDF["Discharge"].resample('AS-OCT').apply(Calc7Q)
    WYDataDF["3xMedian"] = DataDF["Discharge"].resample('AS-OCT').apply(CalcExceed3TimesMedian)
    
    return ( WYDataDF )


# In[14]:


def GetMonthlyStatistics(DataDF):
    """This function calculates monthly descriptive statistics and metrics 
    for the given streamflow time series.  Values are returned as a dataframe
    of monthly values for each year."""
 
    # resampling the data  
    MoDataDF = DataDF.resample('MS').mean()
    
    MoDataDF['site_no'] =  DataDF['site_no'][0]
    MoDataDF["Mean Flow"] = DataDF["Discharge"].resample('MS').mean()
    MoDataDF["Coeff Var"] = DataDF["Discharge"].resample('MS').std() / MoDataDF["Mean Flow"] * 100.
    MoDataDF["Tqmean"] = DataDF["Discharge"].resample('MS').apply(CalcTqmean)
    MoDataDF["R-B Index"] = DataDF["Discharge"].resample('MS').apply(CalcRBindex)

    return ( MoDataDF )


# In[11]:


def GetAnnualAverages(WYDataDF):
    """This function calculates annual average values for all statistics and
    metrics.  The routine returns an array of mean values for each metric
    in the original dataframe."""
    
    # calculating the annual averages
    AnnualAverages=WYDataDF.mean(axis=0)
        
    return( AnnualAverages )


# In[12]:


def GetMonthlyAverages(MoDataDF):
    """This function calculates annual average monthly values for all 
    statistics and metrics.  The routine returns an array of mean values 
    for each metric in the original dataframe."""
    
    # selecting months
    Months = MoDataDF.index.month
    
    # creating dataframe containing means 
    MonthlyAverages = MoDataDF.groupby(Months).mean()
    
    return( MonthlyAverages )


# In[21]:


# the following condition checks whether we are running as a script, in which 
# case run the test code, otherwise functions are being imported so do not.
# put the main routines from your code after this conditional check.

if __name__ == '__main__':

    # define filenames as a dictionary
    # NOTE - you could include more than jsut the filename in a dictionary, 
    #  such as full name of the river or gaging site, units, etc. that would
    #  be used later in the program, like when plotting the data.
    fileName = { "Wildcat": "WildcatCreek_Discharge_03335000_19540601-20200315.txt",
                 "Tippe": "TippecanoeRiver_Discharge_03331500_19431001-20200315.txt" }
    
    # define blank dictionaries (these will use the same keys as fileName)
    DataDF = {}
    MissingValues = {}
    WYDataDF = {}
    MoDataDF = {}
    AnnualAverages = {}
    MonthlyAverages = {}

    # process input datasets
    for file in fileName.keys():
        
        print( "\n", "="*50, "\n  Working on {} \n".format(file), "="*50, "\n" )
        
        DataDF[file], MissingValues[file] = ReadData(fileName[file])
        print( "-"*50, "\n\nRaw data for {}...\n\n".format(file), DataDF[file].describe(), "\n\nMissing values: {}\n\n".format(MissingValues[file]))
        
        # clip to consistent period
        DataDF[file], MissingValues[file] = ClipData( DataDF[file], '1969-10-01', '2019-09-30' )
        print( "-"*50, "\n\nSelected period data for {}...\n\n".format(file), DataDF[file].describe(), "\n\nMissing values: {}\n\n".format(MissingValues[file]))
        
        # calculate descriptive statistics for each water year
        WYDataDF[file] = GetAnnualStatistics(DataDF[file])
        
        # calcualte the annual average for each stistic or metric
        AnnualAverages[file] = GetAnnualAverages(WYDataDF[file])
        
        print("-"*50, "\n\nSummary of water year metrics...\n\n", WYDataDF[file].describe(), "\n\nAnnual water year averages...\n\n", AnnualAverages[file])

        # calculate descriptive statistics for each month
        MoDataDF[file] = GetMonthlyStatistics(DataDF[file])

        # calculate the annual averages for each statistics on a monthly basis
        MonthlyAverages[file] = GetMonthlyAverages(MoDataDF[file])
        
        print("-"*50, "\n\nSummary of monthly metrics...\n\n", MoDataDF[file].describe(), "\n\nAnnual Monthly Averages...\n\n", MonthlyAverages[file])
                


# In[25]:


#------------------ Output Files --------------------# 
# Annual and Monthly Metrics csv output
Ann_WY = pd.concat([WYDataDF['Wildcat'],WYDataDF['Tippe']])
    ## Annual Wildcat 
Ann_WY.loc[(Ann_WY.site_no == 3335000),'Station'] = 'Wildcat'
    ## Annual Tippecanoe
Ann_WY.loc[(Ann_WY.site_no == 3331500),'Station'] = 'Tippecanoe'
Ann_WY.to_csv('Annual_Metrics.csv', sep=',',index=True)

# Monthly
Mo_WY = pd.concat([MoDataDF['Wildcat'],MoDataDF['Tippe']])
    ## Monthly Wildcat
Mo_WY.loc[(Mo_WY.site_no == 3335000),'Station'] = 'Wildcat'
    ## Monthly Tippecanoe
Mo_WY.loc[(Mo_WY.site_no == 3331500),'Station'] = 'Tippecanoe'
Mo_WY.to_csv('Monthly_Metrics.csv', sep=',',index=True)

# Annual and monthly Average Metrics csv output
    ## Wildcat and Tippecanoe Average Annual
## Annual Average
Mean_Ann = pd.concat([pd.Series('Tippecanoe',index=['Station']).append(AnnualAverages['Tippe']),pd.Series('Wildcat',index=['Station']).append(AnnualAverages['Wildcat'])])
Mean_Ann.to_csv('Average_Annual_Metrics.txt', sep='\t',index=True)
## Month Average
Mean_Mo = pd.concat([MonthlyAverages['Tippe'],MonthlyAverages['Wildcat']])
Mean_Mo.loc[(Mean_Mo.site_no == 3335000),'Station'] = 'Wildcat'
Mean_Mo.loc[(Mean_Mo.site_no == 3331500),'Station'] = 'Tippecanoe'
Mean_Mo.to_csv('Average_Monthly_Metrics.txt', sep='\t',index=True)






