# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 09:05:47 2020

@author: Micah Gross

"""
# initiate app in Anaconda Navigator with
# cd "C:\Users\BASPO\.spyder-py3\Quantum"
# cd "C:\Users\BASPO\.spyder-py3\streamlit_apps\upper_body"
# streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
# import xlsxwriter
# import openpyxl
# import xlwings
from io import BytesIO
import base64

def dfDict_to_json(dict_of_df, orient, **kwargs):# kwargs e.g., date_format='iso' # dict_of_df = Laps
    jsonified_dict={}
    for key in dict_of_df.keys():# key = list(dict_of_df.keys())[0]
        if 'date_format' in kwargs:
            jsonified_dict[(int(key) if 'int' in str(type(key)) else key)]=dict_of_df[key].to_json(orient=orient, date_format=kwargs.get('date_format'))
        else:
            jsonified_dict[(int(key) if 'int' in str(type(key)) else key)]=dict_of_df[key].to_json(orient=orient)
    return jsonified_dict# dict of jsonified DataFrames

def regular_time(duration,increment=0.003,orig_time_and_signal=None):
    '''
    Parameters
    ----------
    duration : float
        DESCRIPTION. e.g. round(np.sum(L_SampleDuration),3)
    increment : float, optional
        DESCRIPTION. e.g. 0.003
    orig_time_and_signal : list, optional
        DESCRIPTION. e.g. [np.array([np.sum(L_SampleDuration[0:i]) for i in range(1,1+len(L_SampleDuration))]),L_Position]

    Returns
    -------
    time : pd.Series
        DESCRIPTION. time with regular increment
    signal : pd.Series, optional
        DESCRIPTION. signal corresponding to regular time series

    '''
    if type(increment)==float:# if increment is a float
        rounder=str(increment)[::-1].find('.')
        duration=round(duration,rounder)
        if round(round(duration/increment,0)*increment,rounder)==duration:# if duration is perfectly divisible by increment
            t=np.linspace(0,duration,1+int(round(duration/increment,0)))
        else:# if duration is not divisible by increment
            t=np.linspace(0,int(duration/increment)*increment,1+int(duration/increment))
    elif type(increment)==int:
        if int(duration/increment)*increment==duration:# if duration is perfectly divisible by increment
            t=np.linspace(0,duration,1+int(duration/increment))
        else:# if duration is not divisible by increment
            t=np.linspace(0,int(duration/increment)*increment,1+int(duration/increment))

    if orig_time_and_signal==None:
        return pd.Series(t)
    else:
        StandardizedSignal=np.interp(t,orig_time_and_signal[0],orig_time_and_signal[1])
        return [pd.Series(t),pd.Series(StandardizedSignal)]

def F_calc_from_impulse(regular_phase_signals, programmed_resistance, side_load_mass):
    '''
    Parameters
    ----------
    regular_phase_signals : TYPE
        DESCRIPTION.
    programmed_resistance : TYPE
        DESCRIPTION.
    side_load_mass : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    weight=(programmed_resistance+side_load_mass)*9.81
    a=regular_phase_signals['Acceleration [m/(s^2)]']
    F_machine=regular_phase_signals['Force [N]']
    F_net=side_load_mass*a+weight
    F_calc=F_net-F_machine
    return F_calc

# def calculate_parameters(regular_phase_signals,machine,a_threshold=0.0):
#     on=regular_phase_signals[machine][regular_phase_signals[machine]['Acceleration [m/(s^2)]']>a_threshold].index.min()
#     off=regular_phase_signals[machine][regular_phase_signals[machine]['Acceleration [m/(s^2)]']>a_threshold].index.max()
#     return pd.DataFrame({'Fpeak':regular_phase_signals[machine].loc[on:off,'Force [N]'].max(),
#                          'Fmean':regular_phase_signals[machine].loc[on:off,'Force [N]'].mean(),
#                          'Fpeak_calc':regular_phase_signals[machine].loc[on:off,'F_calc [N]'].max(),
#                          'Fmean_calc':regular_phase_signals[machine].loc[on:off,'F_calc [N]'].mean(),
#                          'vpeak':regular_phase_signals[machine].loc[on:off,'Speed [m/s]'].max(),
#                          'vmean':regular_phase_signals[machine].loc[on:off,'Speed [m/s]'].mean(),
#                          'Ppeak':regular_phase_signals[machine].loc[on:off,'Power [W]'].max(),
#                          'Pmean':regular_phase_signals[machine].loc[on:off,'Power [W]'].mean(),
#                          'Ppeak_calc':regular_phase_signals[machine].loc[on:off,'P_calc [W]'].max(),
#                          'Pmean_calc':regular_phase_signals[machine].loc[on:off,'P_calc [W]'].mean(),
#                          'distance':regular_phase_signals[machine].loc[on:off,'Position [m]'].max()-regular_phase_signals[machine].loc[on:off,'Position [m]'].min(),# max position minus min position
#                          'duration':regular_phase_signals[machine].loc[on:off,'Time [s]'].max()-regular_phase_signals[machine].loc[on:off,'Time [s]'].min()},index=[0])
def calculate_parameters(regular_phase_signals,machine):
    return pd.DataFrame({'Fpeak':regular_phase_signals[machine]['Force [N]'].max(),
                         'Fmean':regular_phase_signals[machine]['Force [N]'].mean(),
                         'Fpeak_calc':regular_phase_signals[machine]['F_calc [N]'].max(),
                         'Fmean_calc':regular_phase_signals[machine]['F_calc [N]'].mean(),
                         'vpeak':regular_phase_signals[machine]['Speed [m/s]'].max(),
                         'vmean':regular_phase_signals[machine]['Speed [m/s]'].mean(),
                         'Ppeak':regular_phase_signals[machine]['Power [W]'].max(),
                         'Pmean':regular_phase_signals[machine]['Power [W]'].mean(),
                         'Ppeak_calc':regular_phase_signals[machine]['P_calc [W]'].max(),
                         'Pmean_calc':regular_phase_signals[machine]['P_calc [W]'].mean(),
                         'distance':regular_phase_signals[machine]['Position [m]'].max()-regular_phase_signals[machine]['Position [m]'].min(),# max position minus min position
                         'duration':regular_phase_signals[machine]['Time [s]'].max()-regular_phase_signals[machine]['Time [s]'].min()},index=[0])

def results_parameters_signals(file_dfs, df_setting, name, date, exercises, side_loads):#, parameters_to_excel=False, signals_to_excel=False):
    # prepare dictionaries
    Results_signals={}# raw signals
    Results_parameters={}# parametrized results
    exercise_file_container_1 = st.empty()
    exercise_file_container_2 = st.empty()
    load_rep_container = st.empty()
    # iterate through exercises
    for exercise in exercises:# exercise = exercises[0]
        Results_signals[exercise]={}
        Results_parameters[exercise]={}
        # determine the cable's direction for the concentric movment
        if exercise=='Syncro smith bilateral':
            conc_direction='in'
        elif exercise=='Syncro bilateral':
            conc_direction='out'
        exercise_file_names = list(df_setting[df_setting['Exercise']==exercise]['File_name'])
        for f_nr,f_name in enumerate(exercise_file_names):# f_nr,f_name = 0,exercise_file_names[0]
            df_in = file_dfs[f_name]
            # print which file is being processed
            exercise_file_container_1.text('currently processing file '+str(
                df_setting[df_setting['File_name']==f_name].index[0]+1
                )+' of '+str(len(df_setting)))
            exercise_file_container_2.text(exercise+' file nr '+str(f_nr+1)+': / '+f_name)
    
            # generate appropriate keys for Results_signals and Results_parameters
            Results_signals[exercise]['file_'+str(f_nr+1)]={}
            # Results_parameters[exercise]['file_'+str(f_nr+1)]={}
    
            # iterate through reps
            for rep in df_in['Repetition # (per Set)'].unique():# rep=1 # rep=2
                # generate appropriate keys for Results_signals and Results_parameters
                Results_signals[exercise]['file_'+str(f_nr+1)]['Rep'+str(rep)]={}
                # Results_parameters[exercise]['file_'+str(f_nr+1)]['rep_'+str(rep)]={}
    
                if 'summary' not in Results_parameters[exercise].keys():# true for first running rep only
                    Results_parameters[exercise]['summary']=pd.DataFrame()
                    running_rep=1# total running rep number within exercise
                else:
                    running_rep=Results_parameters[exercise]['summary'][Results_parameters[exercise]['summary']['f_nr']>0].index[-1]+1# total running rep number within exercise
                    
                if exercise=='Syncro smith bilateral':
                    col='Eccentric/Assisted load [kg]'
                    mo_type='Ecc. / Assist.'
                    programmed_resistance=float(df_in[((df_in['Repetition # (per Set)']==rep) & (df_in['Motion type']==mo_type))][col].mode())# programmed motor resistance for the eccentric phase in kg
                    load_nr = f_nr+1
                elif exercise=='Syncro bilateral':
                    col='Concentric/Resisted load [kg]'
                    mo_type='Con. / Resist.'
                    try:# in case the concentric phase is indeed missing
                        programmed_resistance=float(df_in[((df_in['Repetition # (per Set)']==rep) & (df_in['Motion type']==mo_type))][col].mode())# programmed motor resistance for the concentric phase in kg
                        load_nr = int(pd.Series([abs(programmed_resistance-ld) for ld in side_loads[exercise]],index=side_loads[exercise].index).idxmin())
                    except:
                        continue
    
                load_mass=2*side_loads.loc[load_nr,exercise]
                # display load for monitoring purposes
                load_rep_container.text('load '+str(load_nr)+', rep '+str(rep)+' ('+str(load_mass)+' kg)')

                # iterate through phases
                for phase in ['out','in']:# phase='out' # phase='in'
                    # generate appropriate keys for Results_signals and Results_parameters
                    # Results_signals[exercise]['file_'+str(f_nr+1)]['Rep'+str(rep)][phase]={}
                    # Results_parameters[exercise]['file_'+str(f_nr+1)]['rep_'+str(rep)][phase]={}
                    
                    # create dict to store signals within phase
                    phase_signals={}
                    regular_phase_signals={}
                    
                    # extract desired raw signals
                    for machine in [1,2]:# machine=1
                        phase_signals[machine]=df_in[((df_in['Repetition # (per Set)']==rep) & (df_in['Machine index (In Syncro)']==machine) & (df_in['Motion type']=='Con. / Resist.' if phase=='out' else df_in['Motion type']=='Ecc. / Assist.'))][Options['use_cols']]
                        phase_signals[machine].columns=Options['use_cols']# phase_signals[machine][['Speed [m/s]', 'Acceleration [m/(s^2)]']].plot()
                        if phase==conc_direction:
                            i_end_prop = list(np.sign(phase_signals[machine]['Speed [m/s]']) == np.sign(phase_signals[machine]['Acceleration [m/(s^2)]'])).index(False)
                            phase_signals[machine] = phase_signals[machine].iloc[:i_end_prop]# phase_signals[machine][['Speed [m/s]', 'Acceleration [m/(s^2)]']].plot()
                            # # i_end_prop = max(np.where(np.sign(phase_signals[machine]['Speed [m/s]']) == np.sign(phase_signals[machine]['Acceleration [m/(s^2)]']))[0])
                            # idx_end_prop = phase_signals[machine].index[i_end_prop]
                            # phase_signals[machine] = phase_signals[machine].loc[:idx_end_prop]
                            
                    if len(phase_signals[1])*len(phase_signals[2])==0:# if the phase wasn't recorded for this rep
                        st.write('rep '+str(rep)+', '+phase+': no available data')
                        continue
                
                    # change the direction of speed to positive
                    if ((conc_direction=='in') & (phase=='in')):
                        for machine in [1,2]:# machine=1
                            for sig in ['Speed [m/s]','Acceleration [m/(s^2)]']:# sig='Speed [m/s]'
                                phase_signals[machine][sig]=-1*phase_signals[machine][sig]

                    #standardize time and signals
                    for machine in [1,2]:# machine=1
                        regular_phase_signals[machine]=pd.DataFrame()
                        orig_time=phase_signals[machine]['Sample duration [s]'].cumsum()
                        regular_phase_signals[machine]['Time [s]']=regular_time(duration=round(orig_time.max(),3))
                        for n,sig in enumerate([c for c in Options['use_cols'] if c!='Sample duration [s]']):# n,sig=0,'Position' # n,sig=1,'Speed [m/s]'
                            regular_phase_signals[machine][sig]=regular_time(duration=round(orig_time.max(),3),
                                           orig_time_and_signal=[orig_time,
                                                                  phase_signals[machine][sig]])[1]
                    # calculate F_calc 
                    if ((conc_direction=='in') & (phase=='in')):
                        if Options['F_calc_method']=='acceleration':# calculate from acceleration and assumed resistance
                            st.write('Options[F_calc_method]==acceleration not programmed yet')
    
                        if Options['F_calc_method']=='impulse':# calculate using F*dt=m*dv
                            for machine in [1,2]:# machine=1
                                regular_phase_signals[machine]['F_calc [N]']=F_calc_from_impulse(regular_phase_signals[machine], programmed_resistance,
                                                     side_load_mass=0.5*load_mass)
            
                        elif Options['F_calc_method']=='energy':
                            st.write('Options[F_calc_method]==energy not programmed yet')
            
                        else:
                            st.write('no F_calc method specified')
                                    
                    else:# if phase or conc_direction is 'out', in which case F_calc is unnecessary
                        for machine in [1,2]:# machine=1
                            regular_phase_signals[machine]['F_calc [N]']=None#np.nan
                        
                    # calculate power
                    for machine in [1,2]:# machine=1
                        regular_phase_signals[machine]['Power [W]']=regular_phase_signals[machine]['Force [N]']*regular_phase_signals[machine]['Speed [m/s]']
                        regular_phase_signals[machine]['P_calc [W]']=regular_phase_signals[machine]['F_calc [N]']*regular_phase_signals[machine]['Speed [m/s]']
                
                    # calculate combined values (both sides)
                    regular_phase_signals['combined']=pd.DataFrame(columns=regular_phase_signals[1].columns)
                    length_combined=np.min([len(regular_phase_signals[1]['Time [s]']),# sample number of machine 1
                                        len(regular_phase_signals[2]['Time [s]'])])# sample number of machine 2 (take the lower of the two)
                    regular_phase_signals['combined']['Time [s]']=regular_phase_signals[1]['Time [s]'][:length_combined]# the shorter of the two sides
                    combined_position=pd.Series(0,index=regular_phase_signals['combined']['Time [s]'].index)
                    for i in combined_position.index[1:]:
                        dx1=regular_phase_signals[1].loc[i,'Position [m]']-regular_phase_signals[1].loc[i-1,'Position [m]']# positional change, machine 1
                        dx2=regular_phase_signals[2].loc[i,'Position [m]']-regular_phase_signals[2].loc[i-1,'Position [m]']# positional change, machine 2
                        dxmean=(dx1+dx2)/2.# positional change, mean of two machines
                        combined_position.loc[i]=combined_position.loc[i-1]+dxmean
                        
                    regular_phase_signals['combined']['Position [m]']=combined_position# based on mean of two machines
                    regular_phase_signals['combined']['Speed [m/s]']=(regular_phase_signals[1]['Speed [m/s]']+regular_phase_signals[2]['Speed [m/s]'])[:length_combined]/2# average of the two sides
                    regular_phase_signals['combined']['Acceleration [m/(s^2)]']=(regular_phase_signals[1]['Acceleration [m/(s^2)]']+regular_phase_signals[2]['Acceleration [m/(s^2)]'])[:length_combined]/2# average of the two sides
                    regular_phase_signals['combined']['Force [N]']=(regular_phase_signals[1]['Force [N]']+regular_phase_signals[2]['Force [N]'])[:length_combined]# sum of the two sides
                    regular_phase_signals['combined']['Power [W]']=(regular_phase_signals[1]['Power [W]']+regular_phase_signals[2]['Power [W]'])[:length_combined]# sum of the two sides
                    regular_phase_signals['combined']['F_calc [N]']=(regular_phase_signals[1]['F_calc [N]']+regular_phase_signals[2]['F_calc [N]'])[:length_combined]# sum of the two sides
                    regular_phase_signals['combined']['P_calc [W]']=(regular_phase_signals[1]['P_calc [W]']+regular_phase_signals[2]['P_calc [W]'])[:length_combined]# sum of the two sides

                    # for machine in [1,2,'combined']:# machine=1 machine='combined'
                    #     Results_parameters[exercise]['file_'+str(f_nr+1)]['rep_'+str(rep)][phase][machine]=calculate_parameters(regular_phase_signals,machine,a_threshold=0.0)

                    # save phase parameters to summary table
                    i=running_rep#-1
                    Results_parameters[exercise]['summary'].loc[i,'f_nr']=f_nr+1
                    Results_parameters[exercise]['summary'].loc[i,'rep_nr']=rep
                    Results_parameters[exercise]['summary'].loc[i,'load_mass']=load_mass
                    Results_parameters[exercise]['summary'].loc[i,'load_nr']=load_nr
                        
                    for machine in [1,2,'combined']:# machine=1 machine='combined'
                        for par in ['Fpeak','Fmean','Fpeak_calc','Fmean_calc','vpeak','vmean','Ppeak','Pmean','Ppeak_calc','Pmean_calc','distance','duration']:# par='Fpeak'
                            Results_parameters[exercise]['summary'].loc[i,phase+'_'+('m'+str(machine) if type(machine)==int else machine)+'_'+par]=calculate_parameters(regular_phase_signals,machine)[par].values[0]

                    # save regular_phase_signals to summary dictionary
                    if phase==conc_direction:
                        Results_signals[exercise]['file_'+str(f_nr+1)]['Rep'+str(rep)][phase]=regular_phase_signals

        # sort the summary table
        if 'summary' in Results_parameters[exercise].keys():
            Results_parameters[exercise]['summary'].sort_values(by=['load_nr','rep_nr'])
    exercise_file_container_1.empty()
    exercise_file_container_2.empty()
    load_rep_container.empty()
    return Results_parameters, Results_signals

def oneLiner(Results_parameters, name, date, body_mass, method='mean', exercises=['Syncro smith bilateral', 'Syncro bilateral']):
    '''
    'method' can be either 'median',  'mean' or 'best'
    method = Options['oneLiner_method']
    '''
    oneLiner=pd.DataFrame()
    oneLiner.loc[0,'subject']=name
    oneLiner.loc[0,'date']=date
    oneLiner.loc[0,'body_mass']=body_mass
    for exercise in exercises:# exercise=exercises[0] # exercise=exercises[exercises.index(exercise)+1]
        try:
            df_ex_sum=Results_parameters[exercise]['summary']
        except:
            df_ex_sum=pd.DataFrame(
                columns=Results_parameters[list(Results_parameters.keys())[0]]['summary'].columns
                )
        for load_nr in range(1,7):# load_nr=1 # load_nr=load_nr+1
            load_nr=int(load_nr)
            df_ex_sum_ld=df_ex_sum[df_ex_sum.load_nr==load_nr]
            for par in df_ex_sum_ld.columns[1]:# par=df_ex_sum_ld.columns[1]
                oneLiner.loc[0,'_'.join([exercise,'n_reps','load'+str(load_nr)])]=len(df_ex_sum_ld['rep_nr'])
            for par in df_ex_sum_ld.columns[2:]:# par=df_ex_sum_ld.columns[3]
                if method=='mean':
                    oneLiner.loc[0,'_'.join([exercise,par,'load'+str(load_nr)])]=df_ex_sum_ld[par].mean()
                elif method=='median':
                    oneLiner.loc[0,'_'.join([exercise,par,'load'+str(load_nr)])]=df_ex_sum_ld[par].median()
                elif method=='best':
                    oneLiner.loc[0,'_'.join([exercise,par,'load'+str(load_nr)])]=df_ex_sum_ld[par].max()
                
    oneLiner=oneLiner[list(oneLiner.columns[0:3])+sorted(list(oneLiner.columns[3:]))]
    oneLiners={}
    for exercise in exercises:# ['Syncro bilateral', 'Syncro smith bilateral']:# exercise = 'Syncro bilateral' # exercise = 'Syncro smith bilateral' 
        oneLiners[exercise]={}   
        cols_ex_full=[col for col in oneLiner.columns[:3]]+[col for col in oneLiner.columns[3:] if exercise in col]
        ex_full=oneLiner[cols_ex_full]# exercise, full
        oneLiners[exercise]['full']=ex_full   
        phase='_in_' if exercise=='Syncro smith bilateral' else '_out_'
        cols_ex_simple=[col for col in ex_full.columns[:3]]+[col for col in ex_full.columns[3:] if (((phase in col) & ('combined' in col)) or ('load_mass' in col) or ('n_reps' in col))]
        ex_simple=ex_full[cols_ex_simple]# exercise, simple
        oneLiners[exercise]['simple']=ex_simple
        try:# if it's not the first iteration of exercise
            columns=list(list(set(list(all_ex_simple.columns)+list(ex_simple.columns))))
            general_cols=list(ex_simple.columns[:3])
            test_parameter_cols=[col for col in columns if col not in general_cols]
            columns=general_cols+sorted(test_parameter_cols)
            all_ex_simple=oneLiner[columns]
        except:# if it's the first iteration of exercise
            all_ex_simple=ex_simple

    oneLiners['all_exercises']={}
    oneLiners['all_exercises']['simple']=all_ex_simple
    oneLiners['all_exercises']['full']=oneLiner
    
    return oneLiners

def generate_excel(Results_signals, Results_parameters, oneLiners, Options):
    '''
    Parameters
    ----------
    Options : dict
        dict containing the keys 'parameters_to_excel', 'signals_to_excel'.

    Returns
    -------
    None.

    '''
    # thanks to https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806/12
    writing_excel_container = st.empty()
    writing_excel_container.text('writing to excel')
    output = BytesIO()
    
    with pd.ExcelWriter(output) as writer:
        if Options['parameters_to_excel']:
            for exercise in Results_parameters.keys():# exercise = list(Results_parameters.keys())[0]
                Results_parameters[exercise]['summary'].to_excel(writer, sheet_name=exercise+'_summary', index=False)
                # workbook = writer.book
                # worksheet = writer.sheets[exercise+'_summary']
                # fmt_tenth = workbook.add_format({'num_format': '#.#0.0'})
                # worksheet.set_column('F:F', 15, fmt_tenth)
            if Options['all_oneLiners']:
                for exercise in oneLiners.keys():# exercise = list(oneLiners.keys())[0]
                    for entry_type in oneLiners[exercise].keys():# entry_type = list(oneLiners[exercise].keys())[0]
                        if Options['transpose_oneLiner']:
                            oneLiners[exercise][entry_type].set_index('subject').T.to_excel(writer, sheet_name=exercise+'_'+entry_type)
                        else:
                            oneLiners[exercise][entry_type].set_index('subject').to_excel(writer, sheet_name=exercise+'_'+entry_type)
            else:
                exercise = 'all_exercises'# exercise = list(oneLiners.keys())[0]
                entry_type = 'full'# entry_type = list(oneLiners[exercise].keys())[0]
                if Options['transpose_oneLiner']:
                    oneLiners[exercise][entry_type].set_index('subject').T.to_excel(writer, sheet_name=exercise+'_'+entry_type)
                else:
                    oneLiners[exercise][entry_type].set_index('subject').to_excel(writer, sheet_name=exercise+'_'+entry_type)
        if Options['signals_to_excel']:
            for exercise in Results_signals.keys():# exercise = list(Results_signals.keys())[0]
                for file in Results_signals[exercise].keys():# file = list(Results_signals[exercise].keys())[0]
                    for rep in Results_signals[exercise][file].keys():# rep = list(Results_signals[exercise][file].keys())[0]
                        for direction in Results_signals[exercise][file][rep].keys():# direction = list(Results_signals[exercise][file][rep].keys())[0]
                            df = Results_signals[exercise][file][rep][direction]['combined']
                            for k in [x for x in Results_signals[exercise][file][rep][direction].keys() if x!='combined']:# k=[x for x in Results_signals[exercise][file][rep][direction].keys() if x!='combined'][0]
                                df = pd.concat((df, 
                                                pd.DataFrame(
                                                    data=Results_signals[exercise][file][rep][direction][k].values,
                                                    columns=['_'.join([str(k),c]) for c in Results_signals[exercise][file][rep][direction][k].columns]
                                                    )
                                                ),
                                               axis=1)
                            df.to_excel(writer,
                                        # sheet_name='_'.join([exercise.replace('Syncro ',''),file,rep,direction]),
                                        sheet_name='_'.join([exercise.replace(' bilateral',''),file,rep,direction]),
                                        index=False)
        writer.save()
        processed_data = output.getvalue()
        
    b64 = base64.b64encode(processed_data)
    writing_excel_container.empty()
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="Results.xlsx">Download Results as Excel File</a>' # decode b'abc' => abc
    # wb=xlwings.Book()
    # if Options['signals_to_excel']:
    #     for exercise in Results_signals.keys():# exercise = list(Results_signals.keys())[0]
    #         for file in Results_signals[exercise].keys():# file = list(Results_signals[exercise].keys())[0]
    #             for rep in Results_signals[exercise][file].keys():# rep = list(Results_signals[exercise][file].keys())[0]
    #                 try:
    #                     sht=wb.sheets.add('_'.join([exercise.replace('Syncro ',''),file,rep]))
    #                 except:
    #                     sht=wb.sheets('_'.join([exercise.replace('Syncro ',''),file,rep]))
    #                     # out phase
    #                     sht.range('A1').value='direction: out'
    #                     sht.range('A2').value='combined'
    #                     sht.range('A3').options(index=False).value=Results_signals[exercise][file][rep]['out']['combined']
    #                     sht.range('I1').value='direction: out'
    #                     sht.range('I2').value='machine 1'
    #                     sht.range('I3').options(index=False).value=Results_signals[exercise][file][rep]['out'][1]
    #                     sht.range('Q1').value='direction: out'
    #                     sht.range('Q2').value='machine 2'
    #                     sht.range('Q3').options(index=False).value=Results_signals[exercise][file][rep]['out'][2]
    #                     # in phase
    #                     sht.range('Y1').value='direction: in'
    #                     sht.range('Y2').value='combined'
    #                     sht.range('Y3').options(index=False).value=Results_signals[exercise][file][rep]['in']['combined']
    #                     sht.range('AG1').value='direction: in'
    #                     sht.range('AG2').value='machine 1'
    #                     sht.range('AG3').options(index=False).value=Results_signals[exercise][file][rep]['in'][1]
    #                     sht.range('AO1').value='direction: in'
    #                     sht.range('AO2').value='machine 2'
    #                     sht.range('AO3').options(index=False).value=Results_signals[exercise][file][rep]['in'][2]

    # if Options['parameters_to_excel']:
    #     for exercise in Results_signals.keys():# exercise = list(Results_signals.keys())[0]
    #         sht=wb.sheets.add((exercise+'_summary'))
    #         sht.range('A1').options(index=False).value=Results_parameters[exercise]['summary']
    
    # for exercise in oneLiners.keys():# exercise = list(oneLiners.keys())[0]
    #     for entry_type in oneLiners[exercise].keys():# entry_type = list(oneLiners[exercise].keys())[0]
    #         sht=wb.sheets.add(exercise.replace('Syncro ','')+'_'+entry_type)
    #         if Options['transpose_oneLiner']:
    #             sht.range('A1').options(ignore_index=True).value=oneLiners[exercise][entry_type].set_index('subject').T
    #         else:
    #             sht.range('A1').options(ignore_index=True).value=oneLiners[exercise][entry_type].set_index('subject')
    
    # sht=wb.sheets('Tabelle1')
    # sht.name='Info'
    # sht.range('A1').value='body mass (kg)'
    # sht.range('B1').value=body_mass
    # for i,exercise in enumerate(['Syncro bilateral', 'Syncro smith bilateral']):# [line,exercise]=[0,list(Files.keys())[0]]
    #     c=['A', 'D'][i]
    #     sht.range(c+str(2)).value=exercise
    #     sht.range(c+str(3)).value=side_loads[exercise]

#%%
def get_df_setting(uploaded_files):
    sorted_uploaded_file_names = [f.name for f in uploaded_files if f.name=='RawMotionSampleExport.csv'
                                  ] + sorted([f.name for f in uploaded_files if len(f.name)==29]) +sorted([
                                      f.name for f in uploaded_files if len(f.name)==30]) 
    sorted_uploaded_files = [f for f in uploaded_files for fn in sorted_uploaded_file_names if f.name==fn]# for f in [f for f in uploaded_files if f.name=='RawMotionSampleExport.csv']:
    sorted_uploaded_files = []
    for i,fn in enumerate(sorted_uploaded_file_names):
        sorted_uploaded_files.append([f for f in uploaded_files if f.name==fn][0])

    df_setting = pd.DataFrame(columns=['File_nr', 'File_name', 'Client_name', 'Test_date','Exercise'])
    file_dfs = {}
    for i,uploaded_file in enumerate(sorted_uploaded_files):# i,uploaded_file=0,uploaded_files[0] # i,uploaded_file=i+1,uploaded_files[i+1]
        dataframe = pd.read_csv(uploaded_file,delimiter=';')#,usecols=[1,2,3] dataframe = pd.read_csv(os.path.join(path,uploaded_file),delimiter=';',usecols=[1,2,3])
        name = dataframe['Client name'].unique()[0]# a unique name string
        date = dataframe['Session Date & Time [ISO 8601]'].unique()[0]# a date-time value
        exercise  = dataframe['Exercise'].unique()[0]# a exercise string
        df_setting.loc[i,'File_nr'] = i
        df_setting.loc[i,'File_name'] = uploaded_file.name
        df_setting.loc[i,'Client_name'] = name
        df_setting.loc[i,'Test_date'] = date
        df_setting.loc[i,'Exercise'] = exercise
        file_dfs[uploaded_file.name] = dataframe
    return df_setting, file_dfs

def get_settings_data(df_setting):
    if not len(df_setting['Client_name'].unique())==1:
        # raise Exception('Export files in source directory are from multiple clients')
        st.write('Error: Export files in source directory are from multiple clients')
    else:
        name = df_setting['Client_name'].unique()[0]# a unique name string
    if not len(df_setting['Test_date'].unique())==1:
        date = sorted(df_setting['Test_date'].unique())[0].split('T')[0]+' - '+sorted(df_setting['Test_date'].unique())[-1].split('T')[0]# a date-range string
        st.write('Export files in source directory are from multiple dates')
    else:
        date = df_setting['Test_date'].unique()[0].split('T')[0]# a unique date string
    exercises = list(df_setting['Exercise'].unique())
    settings_data = {'Client_name': name,
                      'Test_date': date,
                      'Exercises': ', '.join(exercises)}
    return name, date, exercises, settings_data

def get_side_loads(protocol_file):#
    pf = protocol_file.read()# pf = os.path.join(os.getcwd(),'sample_data','Test_Protocol_Micah Gross_2020-10-16.xlsx')
    body_mass = np.nanmean([pd.read_excel(pf, sheet, header=1, usecols=[1], nrows=1).iloc[0,0] for sheet in ['Press', 'Pull']])
    df = pd.read_excel(pf, None, header=6, usecols=range(12), nrows=6)#.dropna(how='all').set_index('Stage', drop=False)
    df_press = df['Press'].dropna(how='all').set_index('Stage', drop=False)# pd.ExcelFile(protocol_file).parse('Press', header=6, usecols=range(12)).loc[:5].dropna(how='all').set_index('Stage', drop=False)#, nrows=6, skiprows=7
    df_pull = df['Pull'].dropna(how='all').set_index('Stage', drop=False)# pd.ExcelFile(protocol_file).parse('Pull', header=6, usecols=range(12)).loc[:5].dropna(how='all').set_index('Stage', drop=False)#, nrows=6, skiprows=7
    side_loads = pd.DataFrame({
        # 'Stage': df_pull['Stage'],
        'Syncro smith bilateral': df_press['true load side (kg)'],
        'Syncro bilateral': df_pull['true load side (kg)']
        })
    side_loads = side_loads.set_index(pd.Index(np.arange(1,len(side_loads)+1)))
    return body_mass, side_loads

def user_input_options():
    Options = {}
    Options['parameters_to_excel'] =  st.sidebar.checkbox('export summary parameters',
                                                          value=True,
                                                          key='parameters_to_excel')
    Options['signals_to_excel'] =  st.sidebar.checkbox('export raw signals',
                                                          value=False,
                                                          key='signals_to_excel')
    Options['oneLiner_method'] = st.sidebar.selectbox('summary parameter caclulation method', [
        'mean',
        'median',
        'best'],
        0)
    Options['transpose_oneLiner'] =  st.sidebar.checkbox('transpose one-liner (to column)',
                                                          value=True,
                                                          key='transpose_oneLiner')
    Options['all_oneLiners'] =  st.sidebar.checkbox('export all one-liners',
                                                          value=False,
                                                          key='all_oneLiners')
    Options['use_cols'] = ['Position [m]', 'Speed [m/s]', 'Acceleration [m/(s^2)]', 'Force [N]', 'Sample duration [s]']
    Options['F_calc_method'] = 'impulse'
    # st.write(Options)
    return Options
#%%
st.write("""

# Upper Body MLD Test

""")
st.sidebar.header('Export Reporting Options')
Options = user_input_options()
uploaded_files = st.file_uploader("upload csv export files", accept_multiple_files=True)

if len(uploaded_files)>0:
    st.write('uploaded '+str(len(uploaded_files))+' raw data files ')

df_setting, file_dfs = get_df_setting([
    f for f in uploaded_files if 'RawMotionSampleExport' in f.name
    ])

if len(df_setting)>0:
    # st.write('successfully generated df_setting')
    name, date, exercises, settings_data = get_settings_data(df_setting)

    if settings_data is not None:
        st.write(settings_data)
        protocol_file = st.file_uploader('upload test protocol file (xlsx)', accept_multiple_files=False)

        if protocol_file is not None:
            # st.write('successfully uploaded protocol file')
            body_mass, side_loads = get_side_loads(protocol_file)# protocol_file = uploaded_files[uploaded_files_names.index([n for n in uploaded_files_names if 'Test_Protocol' in n][0])]
            if side_loads is not None:
                # st.write('successfully generated side_loads')
                # body_mass=side_loads['Syncro bilateral'].loc[1]*20# because the first load should be 10% of body_mass i.e., 5% of body_mass per side
                Results_parameters, Results_signals = results_parameters_signals(file_dfs,
                                                                                  df_setting,
                                                                                  name,
                                                                                  date,
                                                                                  exercises,
                                                                                  side_loads,
                                                                                  )
            if Results_parameters is not None:
                oneLiners=oneLiner(Results_parameters, name, date, body_mass, method=Options['oneLiner_method'])
                if oneLiners is not None:
                    if any([Options['parameters_to_excel'], Options['signals_to_excel']]):
                        # generate_excel(Results_signals, Results_parameters, oneLiners, Options)
                        st.markdown(generate_excel(Results_signals, Results_parameters, oneLiners, Options), unsafe_allow_html=True)
                    
                    # # save variables thus far
                    # df_setting.to_json(os.path.join(os.getcwd(),'saved_variables','df_setting.json'), orient='index', date_format='iso')
                    # with open(os.path.join(os.getcwd(),'saved_variables','file_dfs.json'), 'w') as fp:
                    #     json.dump(dfDict_to_json(file_dfs, orient='index'), fp)
                    # with open(os.path.join(os.getcwd(),'saved_variables','name.json'), 'w') as fp:
                    #     json.dump(name, fp)
                    # with open(os.path.join(os.getcwd(),'saved_variables','date.json'), 'w') as fp:
                    #     json.dump(date, fp)
                    # with open(os.path.join(os.getcwd(),'saved_variables','exercises.json'), 'w') as fp:
                    #     json.dump(exercises, fp)
                    # with open(os.path.join(os.getcwd(),'saved_variables','settings_data.json'), 'w') as fp:
                    #     json.dump(settings_data, fp)
                    # side_loads.to_json(os.path.join(os.getcwd(),'saved_variables','side_loads.json'), orient='index', date_format='iso')
                    # with open(os.path.join(os.getcwd(),'saved_variables','body_mass.json'), 'w') as fp:
                    #     json.dump(body_mass, fp)
                    # # with open(os.path.join(os.getcwd(),'saved_variables','Options.json'), 'w') as fp:
                    # #     json.dump(Options, fp)
                    # # with open(os.path.join(os.getcwd(),'saved_variables','uploaded_files_names.json'), 'w') as fp:
                    # #     json.dump([f.name for f in uploaded_files], fp)
                    # # for f in uploaded_files:
                    # #     with open(os.path.join(os.getcwd(),'saved_variables',(f.name.split('.')[0])+'_bytesIO.txt'), 'wb') as fp:
                    # #         fp.write(f.getbuffer())
                    # # for f in protocol_file:
                    # #     # with open(os.path.join(os.getcwd(),'saved_variables',(f.name.split('.')[0])+'_bytesIO.txt'), 'wb') as fp:
                    # #     with open(os.path.join(os.getcwd(),'saved_variables','protocol_file_bytesIO.txt'), 'wb') as fp:
                    # #         fp.write(f.getbuffer())
                    # # with open(os.path.join(os.getcwd(),'saved_variables','name_date_exercises.json'), 'w') as fp:
                    # #     json.dump([name, date, exercises], fp)
                    st.write('succeeded!')
        
