#---------------------Section 1: Libraries----------------------#

from dateutil import parser
from tkcalendar import DateEntry
from datetime import date, datetime, timedelta
from tkinter.constants import END, INSERT, LEFT, RIGHT, WORD, Y, W
from tkinter import messagebox
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import tkinter as tk
import os as os
from fpdf import FPDF
import csv as csv

#---------------------Section 2: Mathematic Model---------------#

def parameter_estimation_model(data,t=0):
    """
    parameter_estimation_model calculates paramters alpha_1 and beta_1 of model A using the data provided
    To calculate the parameters, the method uses the forward difference formula
    to approximate the diferential equation of the model.
    :data: closing stock price array or dataframe
    :param t: integer time number at which the model parameters are needed to be calculated.
    :return: list with value of parameters alpha_1, beta_1 and the stock value at time t (So).
    """
    
    try:    
        S=data[t:3+t].to_numpy()
    except:
        S=data[t:3+t,0:2];

    s0, s1, s2 = S[0,1], S[1,1], S[2,1]
     
    try:
        beta_1=(s0*s0*(s2 - s1) - s1*s1*s1 + s0*s1*s1)/(s0 * s1 * (s0 - s1))
    except ZeroDivisionError:
        beta_1=(s0*s0*(s2 - s1) - s1*s1*s1 + s0*s1*s1)/(s0 * s1)

    try:
        alpha_1=(s1*s1 - s0*s2)/(s0 * s1 * (s0 - s1))
    except ZeroDivisionError:
        alpha_1=(s1*s1 - s0*s2)/(s0 * s1)

    return alpha_1,beta_1,s0

def model_A(dataModel):
    """
    model A generates and stores the values alpha_1 and beta_1 of model A for each time within the archive data_model
    To recreate the plots of the article regarding the values of the parameters, this parameters are calculated for the length of data model
    The function prints in console the mean value of each of the parameters in time.
    :return: list with value of time, alpha_1, beta_1 and S(t) the stock value at time t (So).
    """
    dataSize = len(dataModel)-2
    parameters=np.zeros([dataSize,4])
    
    for t in range(dataSize):     
        alpha_1,beta_1,So=parameter_estimation_model(dataModel,t)
        parameters[t,0]=t
        parameters[t,1]=alpha_1
        parameters[t,2]=beta_1
        parameters[t,3]=So

    print(r'Mean Values: alpha_1= %f, beta_1= %f'%(np.mean(parameters[:,1]),np.mean(parameters[:,2])))
    
    return parameters

def model_fit(dataModel, parameters):
    """
    -model_fit takes parameters alpha_1 and beta_1 and the solutions of the model to fit actual price data.
    -This function considers the limitations over the 10% stock price fluctuation (ARA and ARB)
    -To choose wich solution equation to use for calculating S(t), when two or more equations are valid, maximum value is chosen.
    :parameters: list of values [time, alpha_1, beta_1 and actual S(t)].
    :return: Array with the fitted values of the stock price and times t, with fit MAPE and RMSPE
    
    """ 

    t_total=np.shape(parameters)[0];
    S_theory=np.zeros([t_total,2])
    for t in range(1,t_total):
        alpha_1,beta_1=parameters[t,1:3];
        So=1
        S_t=So
        if t==1:
            So=parameters[t-1,3];
        else:
            So=S_theory[t-1,1];
            
        to=t-1
        c=np.abs((So+ (beta_1/alpha_1))/So)
        S_t_1=(beta_1/alpha_1)*(1/c)*np.exp(beta_1*(t-to))/(1-(1/c)*np.exp(beta_1*(t-to)))
        S_t_2=-(beta_1/alpha_1)*(1/c)*np.exp(beta_1*(t-to))/(1+(1/c)*np.exp(beta_1*(t-to)))
        S_t_3=(beta_1/alpha_1)*(1/c)*np.exp(-np.abs(beta_1)*(t-to))/(1-(1/c)*np.exp(-np.abs(beta_1)*(t-to)))
        S_t_4=-(beta_1/alpha_1)*(1/c)*np.exp(-np.abs(beta_1)*(t-to))/(1+(1/c)*np.exp(-np.abs(beta_1)*(t-to)))

         
        equations=np.zeros([4]);
        
        if (S_t_1>So and (S_t_1+(beta_1/alpha_1))*S_t_1>0):S_t=S_t_1;equations[0]=1;
        if (S_t_2>So and (S_t_2+(beta_1/alpha_1))*S_t_2<0):S_t=S_t_2;equations[1]=1;
        if (S_t_3<So and (S_t_3+(beta_1/alpha_1))*S_t_3>0):S_t=S_t_3;equations[2]=1;
        if (S_t_4<So and (S_t_4+(beta_1/alpha_1))*S_t_4<0):S_t=S_t_4;equations[3]=1;

        if np.sum(equations)>1:
            S_t_possible=[S_t_1,S_t_2,S_t_3,S_t_4]
            S_t = max(S_t_possible)

          
        if(np.abs((S_t-So)/So)>0.1):S_t=(1+0.1*np.sign(S_t-So))*So

        S_theory[t,0]=t;
        S_theory[t,1]=S_t;

    MAPE,RMSPE=MAPE_n_RMSPE(S_theory,dataModel)
    return S_theory, MAPE, RMSPE

def model_forecast(dataModel,t_start = 0,t_total=30):
    """
    model_forecast forecast the behavior of the stock prices for a given time t using certain initial stock values 
    -This function considers the limitations over the 10% stock price fluctuation (ARA and ARB)
    -To choose wich solution equation to use for calculating S(t), when two or more equations are valid, maximum value is chosen.
    -Parameter value alpha_1 and beta_1 are assumed to be constant only for four days period.
    -S(t=0) to S(t=3) are used to calculate alpha_1 and beta_1 using forward difference method.
    -S(t=4) are calculated by substituting previously obtained alpha_1 and beta_1 and S_0 to S(t).
    -The function prints in console the MAPE and RMSPE of the forecast with respect to real price data.
    :dataModel: list of stock data prices with which the forecast is going to be initialized.
    :t_start: starting day for forecast.
    :t_total: total time in days to forecast.
    :return: Array with the forecasted values of the stock price and times t_total, alongside forecast MAPE and RMSPE
    
    """
    S_theory=np.zeros([t_total+3,2])
    data_init=dataModel[t_start:t_start+4].to_numpy()
    S_theory[0,1]=data_init[0,1]
    S_theory[1,0]=1
    S_theory[1,1]=data_init[1,1]
    S_theory[2,0]=2
    S_theory[2,1]=data_init[2,1]
    
    for to in range(0,t_total):
        alpha_1,beta_1,So=parameter_estimation_model(S_theory,to)
        S_t=So
        t=to+3
        c=np.abs((So+(beta_1/alpha_1))/So)

        S_t_1=(beta_1/alpha_1)*(1/c)*np.exp(beta_1*(t-to))/(1-(1/c)*np.exp(beta_1*(t-to)))
        S_t_2=-(beta_1/alpha_1)*(1/c)*np.exp(beta_1*(t-to))/(1+(1/c)*np.exp(beta_1*(t-to)))
        S_t_3=(beta_1/alpha_1)*(1/c)*np.exp(-beta_1*(t-to))/(1-(1/c)*np.exp(-beta_1*(t-to)))
        S_t_4=-(beta_1/alpha_1)*(1/c)*np.exp(-beta_1*(t-to))/(1+(1/c)*np.exp(-beta_1*(t-to)))

        equations=np.zeros([4])
        
        if (S_t_1>So and (S_t_1+(beta_1/alpha_1))*S_t_1>0):S_t=S_t_1;equations[0]=1;
        if (S_t_2>So and (S_t_2+(beta_1/alpha_1))*S_t_2<0):S_t=S_t_2;equations[1]=1;
        if (S_t_3<So and (S_t_3+(beta_1/alpha_1))*S_t_3>0):S_t=S_t_3;equations[2]=1;
        if (S_t_4<So and (S_t_4+(beta_1/alpha_1))*S_t_4<0):S_t=S_t_4;equations[3]=1;
        
        if np.sum(equations)>1:
            S_t_possible=[S_t_1,S_t_2,S_t_3,S_t_4]
            S_t = max(S_t_possible)

        if(np.abs((S_t-S_theory[t-1,1])/S_theory[t-1,1])>0.1):S_t=(1+0.1*np.sign(S_t-S_theory[t-1,1]))*S_theory[t-1,1]

        #print([t,S_t])
        S_theory[t,0]=t;
        S_theory[t,1]=np.abs(S_t);
        
    MAPE,RMSPE=MAPE_n_RMSPE(S_theory,dataModel[t_start:t_start+t_total+3])
    return S_theory, MAPE, RMSPE

def MAPE_n_RMSPE(S_theory,S_real):
    '''
    MAPE_n_RMSPE calculates the error of the generated data (theoretical and forecast) with respect to real data.
    :param S_theory: array of theoretical stock prices and time
    :param S_real: array of actual stock prices and time
    :return: List with errors MAPE and RMSPE
    '''
    N=np.size(S_theory[1:,0]);
    
    S_real=S_real.to_numpy();
    sumMAPE=0
    sumRMSPE=0
    
    try:
        for i in range(1,N):
            sumMAPE+=np.abs((S_theory[i,1]-S_real[i,1])/S_real[i,1])
            sumRMSPE+=((S_theory[i,1]-S_real[i,1])/S_real[i,1])**2
        
        MAPE=(1/N) * sumMAPE * 100
        RMSPE=np.sqrt((1/N) * sumRMSPE) * 100
    
    except IndexError:
        MAPE,RMSPE = "N/A","N/A"
    return MAPE,RMSPE

#---------------------Section 3: Run Code-----------------------#
"""
In this section of the code, functions to call and utilize the models are called
function run_simulation will run mathematical models given a .csv file from GUI and return plots to display
function generate_report will run mathematical models given a .csv file from GUI and return .pdf file to be saved to local files.
"""
    
def run_simulation(filename, forecast_t_start, forecast_t_total, displayFit, displayForecast, printReport, saveCSV = False):
    data_model = pd.read_csv(filename)
    dateList = list(data_model['Date'])

    parameters=model_A(data_model)
    
    fit_model, fit_MAPE, fit_RMSPE = model_fit(data_model, parameters)
    forecast_model, forecast_MAPE, forecast_RMSPE = model_forecast(data_model,forecast_t_start,forecast_t_total)

    #------------------------for generating .csv files-----#
    if saveCSV:
        with open(f"Forecast {os.path.splitext(os.path.basename(filename))[0]} from {dateList[forecast_t_start]} - {forecast_t_total} days.csv","w",newline="") as f:
            N = np.size(forecast_model[:,0])
            header = ['index','date','real_value','theoretical_value','diff']
            writer = csv.writer(f)

            writer.writerow(header)
            for i in range(0,N):
                try:
                    writer.writerow(
                        [
                            i,
                            dateList[forecast_t_start+i],
                            data_model["Close"][forecast_t_start+i],
                            forecast_model[i,1],
                            forecast_model[i,1] - data_model["Close"][forecast_t_start+i]
                        ])
                except IndexError:
                    writer.writerow(
                        [
                            i,
                            "",
                            "",
                            forecast_model[i,1],
                            ""
                        ])
        
        with open(f"Fit {os.path.basename(filename)} - {dateList[0]} to {dateList[-1]}.csv","w",newline="") as f:
            N = np.size(fit_model[:,0])
            header = ['index','date','real_value','theoretical_value','diff']
            writer = csv.writer(f)

            writer.writerow(header)
            for i in range(0,N):
                try:
                    writer.writerow(
                        [
                            i,
                            dateList[i],
                            data_model["Close"][i],
                            fit_model[i,1],
                            fit_model[i,1] - data_model["Close"][i]
                        ])
                except IndexError:
                    writer.writerow(
                        [
                            i,
                            "",
                            "",
                            fit_model[i,1],
                            ""
                        ])
        return



    #Start plotting here
    fig1 = plt.figure()
    fig2 = plt.figure()

    #----------------------- generate forecast plot
    if displayForecast or printReport:
        ax1 = fig1.add_subplot(1,1,1)
        ax1.plot(
            forecast_model[1:,0],
            forecast_model[1:,1],
            label=r'Forecast')

        ax1.plot(
            data_model[forecast_t_start:forecast_t_start+forecast_t_total+3].to_numpy()[0:,0],
            data_model[forecast_t_start:forecast_t_start+forecast_t_total+3].to_numpy()[0:,1],
            label='Data')
        
        ax1.set_xticks(
            data_model[forecast_t_start:forecast_t_start+forecast_t_total+3].to_numpy()[0:,0][::5])
        ax1.set_title(f"Forecast {os.path.splitext(os.path.basename(filename))[0]}\nFrom {dateList[forecast_t_start]}, {forecast_t_total} trading days long\n(MAPE: {forecast_MAPE:.4}; RMSPE: {forecast_RMSPE:.4})")
        ax1.legend()
        ax1.tick_params(axis='both', which='both', labelsize=7, labelbottom = True, labelrotation = 45)

        title = f"Forecast {os.path.splitext(os.path.basename(filename))[0]} From {dateList[forecast_t_start]} - {forecast_t_total} trading days long"
        fig1.canvas.get_default_filename = lambda : '%s.%s' % (title, fig1.canvas.get_default_filetype())
        #fig1.canvas.set_window_title(title)
        fig1.tight_layout()
        fig1.autofmt_xdate()
    #------------------------generate fit plot
    if displayFit or printReport:
        ax2 = fig2.add_subplot(1,1,1)
        ax2.plot(
            fit_model[1:,0],
            fit_model[1:,1],
            label=r'Theory')

        ax2.plot(
            data_model[:].to_numpy()[:,0],
            data_model[:].to_numpy()[:,1],
            label='Data')
        
        ax2.set_xticks(
            data_model[:].to_numpy()[:,0][::20])
        ax2.set_title(f"Fit {os.path.basename(filename)},\n{dateList[0]} to {dateList[-1]}\n(MAPE: {fit_MAPE:.4}; RMSPE: {fit_RMSPE:.4})")
        ax2.legend()
        ax2.tick_params(axis='both', which='both', labelsize=7, labelbottom = True, labelrotation = 45)

        title = f"Fit {os.path.splitext(os.path.basename(filename))[0]} {dateList[0]} to {dateList[-1]}"
        fig2.canvas.get_default_filename = lambda : '%s.%s' % (title, fig2.canvas.get_default_filetype())
        #fig2.canvas.set_window_title(title)
        fig2.tight_layout()
        fig2.autofmt_xdate()   
    #----------------------- display / pdf branch --
    if printReport:
        fitText = [
            "This model is based on stock price estimation and prediction using non-linear ODE based Logistical Equation",
            "Fit takes parameters alpha1 and beta1 and the solutions of the model to fit actual price data, with algorithm as follows:",
            "1. Parameter value alpha1 and beta1 are assumed to be constant only for four days period.",
            "2. 3-days real data are used to calculate alpha1 and beta1 using forward difference method.",
            "3. Theoretical value for the next date are then calculated by substituting previously obtained alpha1 and beta1 and S_0 to S(t)",
            "4. This function considers the limitations over the 10% stock price fluctuation (ARA and ARB)",
            "5. To choose wich solution equation to use for calculating S(t), when two or more equations are valid, maximum value is chosen.",
            "These processes then are repeated for the entire duration of the provided data.",
            "The goals of fitting process is to test the accuracy of analytical solution against real data for shorter period",
            "",
            "-Mean Absolute Percentage Error (MAPE) and Root Mean Square Percentage Error (RMSPE) against real data then are displayed for error and prediction capability measurements. Specifically, MAPE can be interpreted as follows:",
            "< 10: Highly Accurate Model",
            "10 - 20: Good Model",
            "20 - 50: Reasonable Model",
            "above 50: Inaccurate"
            ]

        forecastText = [
            "This model is based on stock price estimation and prediction using non-linear ODE based Logistical Equation",
            "Forecasts the behavior of the stock prices for a given time t using certain initial stock values, with following algorithm:",
            "1. Parameter value alpha1 and beta1 are assumed to be constant only for four days period.",
            "2. As initialization, theoretical value at t=0 to t=3 are set to be equal to real time data at equal date.",
            "3. 3 Previous theoretical values generated from the model are used to calculate alpha1 and beta1 using forward difference method.",
            "4. The next theoretical value are then calculated by substituting previously obtained alpha1 and beta1 and S_0 to S(t)",
            "5. This function considers the limitations over the 10% stock price fluctuation (ARA and ARB)",
            "6. To choose wich solution equation to use for calculating S(t), when two or more equations are valid, maximum value is chosen.",
            "These processes then are repeated for the entire duration of the provided data.",
            "The goals of forecasting process is to test the accuracy of forecasting model against real data for a longer forecasting duration",
            "",
            "-Mean Absolute Percentage Error (MAPE) and Root Mean Square Percentage Error (RMSPE) against real data then are displayed for error and prediction capability measurements. Specifically, MAPE can be interpreted as follows:",
            "< 10: Highly Accurate Model",
            "10 - 20: Good Model",
            "20 - 50: Reasonable Model",
            "above 50: Inaccurate"
            ""
            ]

        forecastFile, fitFile = f"Forecast {os.path.splitext(os.path.basename(filename))[0]}.png", f"Fit {os.path.splitext(os.path.basename(filename))[0]}.png"
        fig1.savefig(forecastFile)
        fig2.savefig(fitFile)

        currTimestamp = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")

        pdf = FPDF(orientation="portrait",unit="mm",format="A4")
        pdf.add_page()

        pdf.set_font("Arial","B",16)
        pdf.cell(40,10,"Stock Price Analysis Report",ln=1)


        pdf.set_font("Arial","",12)
        pdf.cell(40,6,f"Source File: {filename}",ln=1)
        pdf.cell(40,6,f"Date Generated: {currTimestamp}",ln=1)

        pdf.image(fitFile, w=180)
        pdf.set_font("Arial","B",10)
        pdf.cell(40,6,f"Model Review",ln=1)
        pdf.set_font("Arial","",10)
        pdf.multi_cell(180,5,"\n".join(fitText))
        
        pdf.add_page()
        pdf.image(forecastFile, w=180)
        pdf.set_font("Arial","B",10)
        pdf.cell(40,6,f"Model Review",ln=1)
        pdf.set_font("Arial","",10)
        pdf.multi_cell(180,5,"\n".join(forecastText))

        pdf.output(f"{os.path.splitext(os.path.basename(filename))[0]}.pdf")
    
    elif displayForecast:
        #fig1.show()
        return fig1
    
    elif displayFit:
        #fig2.show()
        return fig2

def string_to_date(string):
    return parser.parse(string)

#---------------------Section 4: Analytical, Non-UI-------------#
"""
In this section of the code, automations for analytical, thesis-writing purpose are made. No GUI and data are displayed through terminal output for copy-paste later or plot only for save later.
function forecast_simulation_without_UI -> basic forecast simulation without plot, iterated for every 15 trading days
function multiple_forecast_without_UI -> calling forecast_simulation_without_UI for different durations
function plotting -> calling run_simulation for multiple random period, generating plots every time.
"""

def forecast_simulation_without_UI(filename, forecast_t_total):
    data_model = pd.read_csv(filename)
    dateList = list(data_model['Date'])
    simulationResult = pd.DataFrame(columns=['Date','MAPE','RMSPE'])

    forecast_t_start = 0

    while True:
        try:
            forecast_model, forecast_MAPE, forecast_RMSPE = model_forecast(data_model,forecast_t_start,forecast_t_total)
            simulationResult = simulationResult.append({'Date': dateList[forecast_t_start], 'MAPE': forecast_MAPE, 'RMSPE': forecast_RMSPE}, ignore_index=True)
            forecast_t_start += 15
        
        except IndexError:
            break
    print(f"Simulation on {filename}, {forecast_t_total} days")
    print(simulationResult)

def multiple_forecasts_without_UI(filename):
    print(f"Running simulation on {filename}")
    forecast_simulation_without_UI(filename,30)
    forecast_simulation_without_UI(filename,45)
    forecast_simulation_without_UI(filename,60)

def plotting(filename):
    print(f"Running simulation on {filename}")
    run_simulation(filename,0,30,True,False,False)
    run_simulation(filename,0,30,False,True,False)
    run_simulation(filename,245,45,False,True,False)
    run_simulation(filename,480,60,False,True,False)

#---------------------Section 5: GUI----------------------------#
"""
In this section of the code, GUI are designed
function main_ui -> displays the first welcome display
function browse_files -> open file selection window
function simulation_options_window -> give simulation options to user based on their chosen file
function open_help_window -> tutorials and helps.
"""

def main_ui():
    window = tk.Tk()
    window.title("Stock Price ODE Analysis")
           
    tk.Label(
        text="Stock Price Analysis Using\nNon-Linear Ordinary Differential Equation", 
        font = "Calibri 18 bold",
        padx=20, pady=20).pack()
    tk.Label(
        text="Andree Sulistio Chandra\n1901478396\n\nUniversitas Bina Nusantara\n2021", 
        font = "Calibri 12",
        padx=20, pady=20).pack()
    
    def quit_program():
        window.quit()
        window.destroy()

    tk.Entry()
    tk.Button(text="💹 Start",width=15,height=2,command=browse_files).pack(pady=10,padx=20, side=LEFT)
    tk.Button(text="❓ Help",width=15,height=2,command=open_help_window).pack(pady=10,padx=20, side=LEFT)
    tk.Button(text="❌ Exit",width=15,height=2,command=quit_program).pack(pady=10,padx=20, side=LEFT)

    window.mainloop()

def browse_files():
    isFileCorrect = False
    while not isFileCorrect:
        filename = tk.filedialog.askopenfilename(title = "Select file",filetypes = (("CSV Files","*.csv"),))
        curr = pd.read_csv(filename)
        if list(curr.columns) == ['Date','Close']:
            #simulation_options_window(filename)
            simulation_window(filename)
            isFileCorrect = True
        else:
            messagebox.showerror("FileError","File Error -- file does not match the format required. Make sure it's a 2 columns .csv file with \'Date\' and \'Close\' as headers and try again.")

def simulation_window(filename):
    simulationWindow = tk.Tk()
    simulationWindow.title(f"Running simulation on {os.path.basename(filename)}...")
    curr = pd.read_csv(filename)

    dateList = [string_to_date(x).strftime("%Y-%m-%d") for x in list(curr['Date'])]
    #print(dateList)        
    startDate = string_to_date(dateList[0])
    endDate = string_to_date(dateList[-1])

    startDateDisplay = startDate.strftime("%Y-%m-%d")
    endDateDisplay = endDate.strftime("%Y-%m-%d")

    tk.Label(
            simulationWindow, 
            text=f"Currently processing: {os.path.basename(filename)}\nData Size: {len(curr)} rows\nFit interval: {startDateDisplay} to {endDateDisplay}",
            font = "Calibri 12", justify=LEFT,
            padx=20, pady=5).grid(row=1, columnspan=2,sticky=W)

    tk.Label(
            simulationWindow, 
            text="-----------------------------------------------\nForecast Parameter",
            font = "Calibri 12 bold", justify=LEFT,
            padx=20, pady=5).grid(row=2, columnspan=2,sticky=W)

    tk.Label(
            simulationWindow, 
            text="Forecast start date",
            font = "Calibri 12", justify=LEFT,
            padx=20, pady=5).grid(row=3, column=0,sticky=W)
        
    forecastDate = DateEntry(simulationWindow, 
            mindate=startDate, 
            maxdate=endDate-timedelta(days=5))
    forecastDate.set_date(startDate)
    forecastDate.grid(row=3,column=1,sticky=W)                      

    tk.Label(
            simulationWindow, 
            text="Forecast duration",
            font = "Calibri 12", justify=LEFT,
            padx=20, pady=5).grid(row=4,column=0,sticky=W)
    
    forecastDuration = tk.Entry(
            simulationWindow, width=10)
    forecastDuration.insert(tk.END,45)
    forecastDuration.grid(row=4,column=1,sticky=W)
    
        
    def startSimulation(displayFit, displayForecast, printReport, saveCSV=False):
        curr_date = forecastDate.get_date()
        
        try:
            curr_duration = int(forecastDuration.get())
        except ValueError:
            messagebox.showerror("ValueError","Enter integers only for forecast duration!")
        
        try:
            date_index = dateList.index(curr_date.strftime("%Y-%m-%d"))
        except ValueError:
            messagebox.showerror("ValueError","Date not found in data provided. Note that valid trading days exclude weekends and some exception days.")
        
        return run_simulation(filename,dateList.index(curr_date.strftime("%Y-%m-%d")),
            curr_duration, displayFit, displayForecast, printReport, saveCSV)

    def print_report():
        print("Hello! This function is to save report as .pdf")
        try:
            startSimulation(displayFit = False, displayForecast = False, printReport = True)
            messagebox.showinfo("✅ Report Generated","Report successfully generated.")
        except Exception as e:
            messagebox.showerror("❌ Report Not Generated","Report not generated due to error: {e}")

    def generate_csv():
        print("Hello! This function is to save raw data as .csv")
        try:
            startSimulation(displayFit = False, displayForecast = False, printReport = False, saveCSV = True)
            messagebox.showinfo("✅ Report Generated","Report successfully generated.")
        except Exception as e:
            messagebox.showerror("❌ Report Not Generated","Report not generated due to error: {e}")
        
 
    fitFigure = startSimulation(True, False, False)
    forecastFigure = startSimulation(False, True, False)
    
    canvas1 = FigureCanvasTkAgg(fitFigure, master = simulationWindow)
    canvas1.draw()
    canvas1.get_tk_widget().grid(padx=20, pady=5, row=1, column = 2, rowspan = 18, columnspan = 4)

    canvas2 = FigureCanvasTkAgg(forecastFigure, master = simulationWindow)
    canvas2.draw()
    canvas2.get_tk_widget().grid(padx=20, pady=5, row=19, column = 2, rowspan = 12, columnspan = 4)

    def updateForecast():
        for item in canvas2.get_tk_widget().find_all():
            canvas2.get_tk_widget().delete(item)
        drawNewForecast()
    
    def drawNewForecast():
        forecastFigure = startSimulation(False, True, False)
        canvas2 = FigureCanvasTkAgg(forecastFigure, master = simulationWindow)
        canvas2.draw_idle()
        canvas2.get_tk_widget().grid(padx=20, pady=5, row=19, column = 2, rowspan = 12, columnspan = 4)

    tk.Button(simulationWindow, 
        text="🔎 Run Forecast",
        width=20,height=2,
        padx=20, pady=5,
        command=updateForecast).grid(padx=20, pady=5, row=6,columnspan=2,sticky=W)

    tk.Button(simulationWindow, 
        text="📰 Save report as .pdf",
        width=20,height=2,
        padx=20, pady=5,
        command=print_report).grid(padx=20, pady=5, row=16,columnspan=2,sticky=W)

    tk.Button(simulationWindow, 
        text="📰 Save raw data as .csv",
        width=20,height=2,
        padx=20, pady=5,
        command=generate_csv).grid(padx=20, pady=5, row=17,columnspan=2,sticky=W)

def open_help_window():
    helpText = [
            "Welcome to this application's help page. This page will have a complete tutorial and guide to navigate this application",
            "This is a GUI made for Stock Price Analysis using Non-Linear ODE",
            "",
            "---",
            "1. Starting the application",
            "To start, click \"💹 Start\". You will then be directed to choose a .csv file to be analyzed.",
            "Only .csv files with 2 columns -- \'Date\' & \'Close\' -- are accepted.", 
            "Otherwise, it will show error message showing that your file is not compatible",
            "",
            "---",
            "2. Application Features",
            "After your .csv files containing data is successfully uploaded, there will be several features available to use.",
            "Fit will take the entire data's date interval and start a fitting process.",
            "Forecast will take user input's starting date and duration and start forecasting process",
            "(See section below for further details on fit and forecast methodology)", 
            "",
            "Display Fit & Forecast -> displaying both fit and forecast for given starting date and duration in one display",
            "Display Fit -> displaying fit only",
            "Display Forecast -> displaying forecast for given starting date and duration only",
            "Generate Report -> generating complete fit and forecast report in .pdf format",
            "---",
            "3. Mathematic Model Recap",
            "This application is based on stock price estimation and prediction using non-linear ODE based Logistical Equation",
            "Forecast the behavior of the stock prices for a given time t using certain initial stock values","-This function considers the limitations over the 10% stock price fluctuation (ARA and ARB)",
            "-To choose wich solution equation to use for calculating S(t), when two or more equations are valid, maximum value is chosen.",
            "-Parameter value alpha_1 and beta_1 are assumed to be constant only for four days period.",
            "-S(t=0) to S(t=3) are used to calculate alpha_1 and beta_1 using forward difference method.",
            "-S(t=4) are calculated by substituting previously obtained alpha_1 and beta_1 and S_0 to S(t)","-The function prints in console the MAPE and RMSPE of the forecast with respect to real price data.",
            "",
            "fit takes parameters alpha_1 and beta_1 and the solutions of the model to fit actual price data.",
            "-This function considers the limitations over the 10% stock price fluctuation (ARA and ARB)",
            "-To choose wich solution equation to use for calculating S(t), when two or more equations are valid, maximum value is chosen."
            ]
        
    helpWindow = tk.Tk()
    helpWindow.title("Help Window")
    tk.Label(helpWindow, text="❓ Help Page",font="Calibri 18 bold",padx=20,pady=20).pack() 
        
    text = tk.Text(helpWindow, wrap=WORD, font="Calibri 11")
    text.insert(INSERT, '\n'.join(helpText))
    text.pack(padx=20, pady=20)

main_ui()
