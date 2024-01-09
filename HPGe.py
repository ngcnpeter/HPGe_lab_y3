import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import io
import requests 
from bs4 import BeautifulSoup
import time
from matplotlib.lines import Line2D
import os
import csv
from os import listdir
from scipy import ndimage
from scipy import stats
from scipy import signal

def nndc(nuc):
    '''This function takes in nuclide name nuc as input and 
       obtain nndc dataset of gamma radiation, output as pandas DataFrame'''
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Content-Type': 'application/x-www-form-urlencoded',
        # 'Cookie': 'JSESSIONID=D70C39328A761ABB0BA3274BD146F79E; _ga=GA1.4.404777615.1698689054; _ga_BCK578C8FS=GS1.1.1699970881.3.0.1699970881.0.0.0; _ga_M7918ZRR2Q=GS1.1.1699970881.3.0.1699970881.0.0.0; _gid=GA1.2.165605393.1699970882; _ga_F5YKVFGCQT=GS1.1.1699970895.3.1.1699971054.0.0.0; _ga=GA1.1.404777615.1698689054; _ga_92P36DL448=GS1.4.1699970895.3.1.1699971191.60.0.0',
        'Origin': 'https://www.nndc.bnl.gov',
        'Pragma': 'no-cache',
        'Referer': 'https://www.nndc.bnl.gov/nudat3/indx_dec.jsp',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'same-origin',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'sec-ch-ua': '"Google Chrome";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
    }

    data = {
        'spnuc': 'name',
        'nuc': nuc,
        'z': '',
        'a': '',
        'n': '',
        'zmin': '',
        'zmax': '',
        'amin': '1',
        'amax': '240',
        'nmin': '',
        'nmax': '',
        'evenz': 'any',
        'evena': 'any',
        'evenn': 'any',
        'tled': 'disabled',
        'tlmin': '0',
        'utlow': 'FS',
        'tlmax': '1E10',
        'utupp': 'GY',
        'dmed': 'disabled',
        'dmn': 'ANY',
        'rted': 'disabled',
        'rtn': 'ANY',
        'reed': 'disabled',
        'remin': '0',
        'remax': '10000',
        'ried': 'disabled',
        'rimin': '0',
        'rimax': '200',
        'ord': 'zate',
        'out': 'file',
        'unc': 'stdandard',
        'sub': 'Search',
    }

    response = requests.post('https://www.nndc.bnl.gov/nudat3/dec_searchi.jsp', headers=headers, data=data)
    soup = BeautifulSoup(response.content,'html.parser')
    nndc_df = pd.read_csv(io.StringIO(soup.find('pre').text), sep="\t")
    #rename nndc columns
    nndc_col = {'Rad Ene.  ' : 'nndc_peak_energy','Unc       .2':'BR_err','Unc       ':'E_err','T1/2 (num)        ':'T1/2(s)' } 
    nndc_df.rename(columns = nndc_col,inplace=True)
    #adding branching ratios of gamma rad. from electron capture and other pathways of the same energy
    nndc_df['BR'] = nndc_df.groupby(['nndc_peak_energy'])['Rad Int.  '].transform('sum')/100 #BR = intensity/100
    nndc_df['rounded_energy'] = round(nndc_df['nndc_peak_energy'],ndigits=0)
    nndc_df['nuc'] = nuc
    #nndc_df['BR_err'] = nndc_df['BR_err'].astype(float)/100
    return nndc_df[['nndc_peak_energy','E_err','BR','BR_err','rounded_energy','nuc','T1/2(s)']]
    
def nndc_all(df):
    '''This function takes in the peak dataframes and find all the nuclides identified, 
       then  output a concatenated dataframe of nndc gamma radiation data for each nuclide (obtained from nndc function defined above)'''
    nuc = df['Nuclide'].dropna().drop(df[df['Nuclide']=='Annihilation'].index)
    nuc = list(set([i[0] for i in nuc.str.findall('\w+\d+')])) #unique nuclide names
    df2 = pd.DataFrame()
    for i in nuc:
        nndc_df = nndc(i)
        df2 = pd.concat([df2,nndc_df])
    return df2

def peak_df(filename):
    ''' This function converts the peak data csv into  a pandas dataframe 
       Input: filename - file path of csv
       Output: df2 '''
    df = pd.read_csv(filename,delimiter = ',',header = [0,1]).droplevel(1,axis = 1)
    rename_dict = {' Nuclide':'Nuclide',' Photopeak_Energy':'Photopeak_Energy','      Peak':'Peak_CPS'
    ,' FWHM':'FWHM','  Net_Area':'Net_Area_Count','Reduced':'Reduced_chi2','   Net_Area':'Net_Area_err'}
    #dtype_dict = {'Photopeak_Energy':float,'Centroid':float,'Peak_CPS':float,'FWHM':float,'Net_Area_Count':float,''}
    df.rename(columns = rename_dict,inplace = True)
    df.dropna(subset='Nuclide',inplace = True)
    df.reset_index(inplace=True) #for merging nuc column later
    return df[['Centroid','Nuclide','Photopeak_Energy','Net_Area_Count','Net_Area_err','Peak_CPS','FWHM','Reduced_chi2']]

def peak_df_new(df):
    '''This function adds additional columns to input peak df:
       1. BR -branching ratio by merging the dataframe from nndc_all() with df on rounded energy 
       (due to discrepancies between interspec and nndc photopeak datasets) 
       2. Centroid error
       3. FWHM error
       4. Resolution
       5. Resolution error
       6. Effective activity - CPS over BR
       
       Input: peak dataframe
       Output: peak dataframe with added columns''' 
    nndc_df = pd.read_csv('nndc_df.csv',index_col=0)
    df['rounded_energy'] = round(df['Photopeak_Energy'],ndigits = 0)
    df = df.dropna(subset='Nuclide').drop(df[df['Nuclide']=='Annihilation'].index)
    nuc = df['Nuclide']
    #convert nuclide names into nndc format, e.g. Pa234m -> Pa234
    df['nuc'] =  pd.Series([i[0] for i in nuc.str.findall('\w+\d+')])
    df2 = df.merge(nndc_df,how = 'left' ,on = ['rounded_energy', 'nuc'])
    df2 = df2.groupby('Photopeak_Energy').max()
    df2['Centroid_err'] = df2['FWHM']/2/np.sqrt(2*np.log(2)*df2['Net_Area_Count'])#Centroid error
    df2['FWHM_err'] = df2['Centroid_err']*np.sqrt(np.log(2))*2 #FWHM error 
    df2['Resolution'] = df2['FWHM']/df2['Centroid']
    df2['Resolution_err']= df2['Resolution']*np.sqrt((df2['Centroid_err']/df2['Centroid'])**2+(df2['FWHM_err']/df2['FWHM'])**2)
    df2['Peak_CPS_BR'] = df2['Peak_CPS']/df2['BR']
    df2.dropna(inplace = True)
    return df2.reset_index()

def chi2_prob(fit,data,err,df):
    '''This function calculates chi2 probability for given 
       fit array, data array,err and degrees of freedom 
       Inputs: fit  - array of fitted values
               data - array of original data
               err  - error 
               df   - degree of freedom
       Outputs: chi2 probability, 
       the probability of obtaining a value of minimized chi2 
       equal to the fit value or higher, given df'''
    norm_resid = (fit -data)/err #normalized residual
    chi2 = np.sum(norm_resid**2) #chi-square
    chi2_red = chi2/df
    chi2_p = 1-stats.chi2.cdf(chi2,df) #chi2 probability =  1-cumulative distribution function
    print(f' chi2: {chi2} \n reduced chi2: {chi2_red}\n chi2 p-value: {chi2_p}')
    return chi2_red

#masses for sediment samples 1-4


def live_real_time(path_list):
    live_time = []
    real_time = []
    for filename in path_list: 
        with open(f'Data/Spectra/{filename}',newline = '') as spec_csv:
            spec_reader = csv.reader(spec_csv, delimiter=',')
            for row in spec_reader:
                if row[0] == 'Live Time (s)':
                    live_time.append(float(row[1]))
                if row[0] == 'Real Time (s)':
                    real_time.append(float(row[1]))
    return live_time,real_time

#all spectrum including unused ones
class Spectra:
    def __init__(self):
        spec_path_list = sorted(listdir('Data/Spectra')[1:])
        live_time = []
        real_time = []
        for filename in spec_path_list: 
            with open(f'Data/Spectra/{filename}',newline = '') as spec_csv:
                spec_reader = csv.reader(spec_csv, delimiter=',')
                for row in spec_reader:
                    if row[0] == 'Live Time (s)':
                        live_time.append(float(row[1]))
                    if row[0] == 'Real Time (s)':
                        real_time.append(float(row[1]))

        #list of spectra array
        self.list = [np.loadtxt(f'Data/Spectra/{path}',delimiter = ',',skiprows = 7,unpack = True) for path in spec_path_list]
        #x (Energy) array
        self.x = self.list[0][1] #keV
        #real time and live time storage
        self.live_time = live_time
        self.real_time = real_time

        # summary dataframe of spectra
        total_cps = [np.sum(self.list[i][2])/self.live_time[i] for i in range(len(spec_path_list))]
        self.df = pd.DataFrame({
                    'Filename'  : spec_path_list,
                    'Live time/s' : self.live_time,
                    'Real time/s' : self.real_time,
                    'Total CPS' : total_cps})

#spectra used in analysis   
class Spec:
    def __init__(self,n):
        sample_mass = np.array([0.427-0.0583, 0.5733-0.0576,0.5399-0.0581,0.5585-0.0581 ])
        #n: 0   bg
        #   1-4 sample 1-4
        #   5   iaea
        spec_list = sorted(listdir('Data/Spectra_final'))
        spec_list = [spec_list[0]]+spec_list[2:]+[spec_list[1]] #reorder
        #list of spectrum array (channel array, energy array, count array,)
        self.list = [np.loadtxt(f'Data/Spectra_final/{path}',delimiter = ',',skiprows = 7,unpack = True) for path in spec_list]
        #real time and live time storage

        live_time = []
        real_time = []
        for filename in spec_list: 
            with open(f'Data/Spectra_final/{filename}',newline = '') as spec_csv:
                spec_reader = csv.reader(spec_csv, delimiter=',')
                for row in spec_reader:
                    if row[0] == 'Live Time (s)':
                        live_time.append(float(row[1]))
                    if row[0] == 'Real Time (s)':
                        real_time.append(float(row[1]))
        self.live_time = live_time
        self.real_time = real_time

        #E (Energy) array
        self.E = self.list[n][1] #keV
        

        #renormalization of sample 3 (abnormal)
        cps_arr = lambda arr: np.concatenate(np.array([self.list[i][2][self.E>2640]/self.live_time[i] for i in arr]))
        cps_mean = lambda arr: np.mean(cps_arr(arr)) 
        correct_n = [1,2,4] #sample (normal) numbers
        #scale factor S
        self.S = cps_mean(correct_n)/cps_mean([3])
        #scale factor error S_err (error propagation of standard error on mean)
        self.S_err = self.S*np.sqrt(np.std(cps_arr(correct_n))**2/len(cps_arr(correct_n))/cps_mean(correct_n)**2
        +np.std(cps_arr([3]))**2/len(cps_arr([3]))/cps_mean([3])**2)

        #corrected cps
        self.cps = self.list[n][2]/self.live_time[n]*[1,1,1,self.S,1,1][n]
        #only for samples 1-4
        self.cps_kg = self.list[n][2]/sample_mass[n-1]