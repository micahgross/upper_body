# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 09:05:47 2020

@author: Micah Gross

"""
# initiate app in Anaconda Navigator with
# cd "C:\Users\BASPO\.spyder-py3\Quantum"
# streamlit run upper_body_mld_app.py
# cd "C:\Users\BASPO\.spyder-py3\streamlit_apps\upper_body"
# streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
# import xlsxwriter
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

def calculate_parameters(regular_phase_signals, machine, a_threshold=0.0):
    on = regular_phase_signals[machine][regular_phase_signals[machine]['Acceleration [m/(s^2)]']>a_threshold].index.min()
    off = regular_phase_signals[machine][regular_phase_signals[machine]['Acceleration [m/(s^2)]']>a_threshold].index.max()
    return pd.DataFrame({'Fpeak':regular_phase_signals[machine].loc[on:off,'Force [N]'].max(),
                         'Fmean':regular_phase_signals[machine].loc[on:off,'Force [N]'].mean(),
                         'Fpeak_calc':regular_phase_signals[machine].loc[on:off,'F_calc [N]'].max(),
                         'Fmean_calc':regular_phase_signals[machine].loc[on:off,'F_calc [N]'].mean(),
                         'vpeak':regular_phase_signals[machine].loc[on:off,'Speed [m/s]'].max(),
                         'vmean':regular_phase_signals[machine].loc[on:off,'Speed [m/s]'].mean(),
                         'Ppeak':regular_phase_signals[machine].loc[on:off,'Power [W]'].max(),
                         'Pmean':regular_phase_signals[machine].loc[on:off,'Power [W]'].mean(),
                         'Ppeak_calc':regular_phase_signals[machine].loc[on:off,'P_calc [W]'].max(),
                         'Pmean_calc':regular_phase_signals[machine].loc[on:off,'P_calc [W]'].mean(),
                         'distance':regular_phase_signals[machine].loc[on:off,'Position [m]'].max()-regular_phase_signals[machine].loc[on:off,'Position [m]'].min(),# max position minus min position
                         'duration':regular_phase_signals[machine].loc[on:off,'Time [s]'].max()-regular_phase_signals[machine].loc[on:off,'Time [s]'].min()},index=[0])

def results_parameters_signals(data_export_file, settings, side_loads, Options):#file_dfs, df_setting, name, date, exercises):#, parameters_to_excel=False, signals_to_excel=False):
    # prepare dictionaries
    Results_signals = {}# raw signals
    Results_parameters = {}# parametrized results
    exercise_container = st.empty()
    set_container = st.empty()
    load_rep_container = st.empty()
    # iterate through exercises
    data_export_file.seek(0)
    data_export_df = pd.read_csv(data_export_file, delimiter=';')
    for exercise in data_export_df['Exercise'].unique():# exercise = data_export_df['Exercise'].unique()[0] # exercise = data_export_df['Exercise'].unique()[1]
        df_ex = data_export_df[data_export_df['Exercise']==exercise]
        exercise_container.text('currently processing '+ exercise)
        # generate appropriate keys for Results_signals and Results_parameters
        Results_signals[exercise] = {}
        Results_parameters[exercise] = {}
        # determine the cable's direction for the concentric movment
        if exercise=='Syncro smith bilateral':
            conc_direction = 'in'
            col = 'Eccentric/Assisted load [kg]'
            mo_type = 'Ecc. / Assist.'
        elif exercise=='Syncro bilateral':
            conc_direction = 'out'
            col = 'Concentric/Resisted load [kg]'
            mo_type = 'Con. / Resist.'
        for set_nr,s_nr in enumerate(df_ex['Set #'].unique(), start=1):# set_nr,s_nr = 1,df_ex['Set #'].unique()[0] # set_nr,s_nr = set_nr+1,df_ex['Set #'].unique()[set_nr]
            df_set = df_ex[df_ex['Set #']==s_nr]
            # print which file is being processed
            set_container.text('set '+str(set_nr))
            # generate appropriate keys for Results_signals
            Results_signals[exercise]['set '+str(set_nr)] = {}
            if any([
                    all([exercise=='Syncro smith bilateral',
                         # settings[settings['Exercise']==exercise][settings['Set']==set_nr]['Mode'].iloc[0]=='NFW',
                         settings[((settings['Exercise']==exercise) & (settings['Set']==set_nr))]['Mode'].iloc[0]=='NFW',
                         settings[((settings['Exercise']==exercise) & (settings['Set']==set_nr))]['Concentric/Resisted load [kg]'].iloc[0]<5]),
                    all([exercise=='Syncro bilateral',
                         settings[((settings['Exercise']==exercise) & (settings['Set']==set_nr))]['Concentric/Resisted load [kg]'].iloc[0]<3,
                         settings[((settings['Exercise']==exercise) & (settings['Set']==set_nr))]['Concentric/Resisted speed limit [m/s]'].iloc[0]<1])
                    ]):# do something else, because it's not a ballistic set
                # print(['non-ballistic set: ', exercise, set_nr])
                settings.loc[
                    settings[((settings['Exercise']==exercise) & (settings['Set']==set_nr))].index,
                    'set_type'
                    ] = 'isokinetic'
                for rep in df_set['Repetition # (per Set)'].unique():# rep = df_set['Repetition # (per Set)'].unique()[0]
                    df_rep = df_set[df_set['Repetition # (per Set)']==rep]
                    # generate appropriate keys for Results_signals
                    Results_signals[exercise]['set '+str(set_nr)]['rep'+str(rep)] = {}
                    for phase in ['out']:# phase='out' # phase='in'
                        phase_signals = {}
                        regular_phase_signals = {}
                        
                        # extract desired raw signals
                        for machine in [1,2]:# machine=1
                            phase_signals[machine] = df_rep[((df_rep['Machine index (In Syncro)']==machine) & (df_rep['Motion type']=='Con. / Resist.' if phase=='out' else df_rep['Motion type']=='Ecc. / Assist.'))][Options['use_cols']]
                            phase_signals[machine].columns=Options['use_cols']# phase_signals[machine][['Speed [m/s]', 'Acceleration [m/(s^2)]']].plot()
                        if len(phase_signals[1])*len(phase_signals[2])==0:# if the phase wasn't recorded for this rep
                            print('rep '+str(rep)+', '+phase+': no available data')
                    
                        #standardize time and signals
                        for machine in [1,2]:# machine=1
                            regular_phase_signals[machine] = pd.DataFrame()
                            orig_time = phase_signals[machine]['Sample duration [s]'].cumsum()
                            regular_phase_signals[machine]['Time [s]'] = regular_time(duration=round(orig_time.max(),3))
                            for n,sig in enumerate([c for c in Options['use_cols'] if c!='Sample duration [s]']):# n,sig=0,'Position' # n,sig=1,'Speed [m/s]'
                                regular_phase_signals[machine][sig] = regular_time(duration=round(orig_time.max(),3),
                                               orig_time_and_signal = [orig_time,
                                                                      phase_signals[machine][sig]])[1]
                        # calculate power
                        for machine in [1,2]:# machine=1
                            regular_phase_signals[machine]['Power [W]'] = regular_phase_signals[machine]['Force [N]']*regular_phase_signals[machine]['Speed [m/s]']
                    
                        # calculate combined values (both sides)
                        regular_phase_signals['combined'] = pd.DataFrame(columns=regular_phase_signals[1].columns)
                        length_combined = np.min([len(regular_phase_signals[1]['Time [s]']),# sample number of machine 1
                                            len(regular_phase_signals[2]['Time [s]'])])# sample number of machine 2 (take the lower of the two)
                        regular_phase_signals['combined']['Time [s]'] = regular_phase_signals[1]['Time [s]'][:length_combined]# the shorter of the two sides
                        combined_position = pd.Series(0,index=regular_phase_signals['combined']['Time [s]'].index)
                        for i in combined_position.index[1:]:
                            dx1 = regular_phase_signals[1].loc[i,'Position [m]']-regular_phase_signals[1].loc[i-1,'Position [m]']# positional change, machine 1
                            dx2 = regular_phase_signals[2].loc[i,'Position [m]']-regular_phase_signals[2].loc[i-1,'Position [m]']# positional change, machine 2
                            dxmean = (dx1+dx2)/2.# positional change, mean of two machines
                            combined_position.loc[i] = combined_position.loc[i-1]+dxmean
                            
                        regular_phase_signals['combined']['Position [m]'] = combined_position# based on mean of two machines
                        regular_phase_signals['combined']['Speed [m/s]'] = (regular_phase_signals[1]['Speed [m/s]']+regular_phase_signals[2]['Speed [m/s]'])[:length_combined]/2# average of the two sides
                        regular_phase_signals['combined']['Acceleration [m/(s^2)]'] = (regular_phase_signals[1]['Acceleration [m/(s^2)]']+regular_phase_signals[2]['Acceleration [m/(s^2)]'])[:length_combined]/2# average of the two sides
                        regular_phase_signals['combined']['Force [N]'] = (regular_phase_signals[1]['Force [N]']+regular_phase_signals[2]['Force [N]'])[:length_combined]# sum of the two sides
                        regular_phase_signals['combined']['Power [W]'] = (regular_phase_signals[1]['Power [W]']+regular_phase_signals[2]['Power [W]'])[:length_combined]# sum of the two sides
                        # save regular_phase_signals to summary dictionary
                        Results_signals[exercise]['set '+str(set_nr)]['rep'+str(rep)][phase] = regular_phase_signals
            else:# it is a ballistic set
                settings.loc[
                    settings[((settings['Exercise']==exercise) & (settings['Set']==set_nr))].index,
                    'set_type'
                    ] = 'ballistic'
                if set_nr > side_loads[exercise][~side_loads[exercise].isna()].index.max():
                    continue
                # else:# iterate through reps
                for rep in df_set['Repetition # (per Set)'].unique():# rep = df_set['Repetition # (per Set)'].unique()[0]
                    df_rep = df_set[df_set['Repetition # (per Set)']==rep]
                    # generate appropriate keys for Results_signals
                    Results_signals[exercise]['set '+str(set_nr)]['rep'+str(rep)] = {}
        
                    if 'summary' not in Results_parameters[exercise].keys() or len(Results_parameters[exercise]['summary'])==0:# true for first running rep only
                        Results_parameters[exercise]['summary'] = pd.DataFrame()
                        running_rep = 1# total running rep number within exercise
                    else:
                        running_rep = Results_parameters[exercise]['summary'][Results_parameters[exercise]['summary']['set_nr']>0].index[-1]+1# total running rep number within exercise
                        
                    if exercise=='Syncro smith bilateral':
                        programmed_resistance = float(df_rep[((df_rep['Repetition # (per Set)']==rep) & (df_rep['Motion type']==mo_type))][col].mode())# programmed motor resistance for the eccentric phase in kg
                        load_nr = set_nr
                        load_mass = 2*side_loads.loc[load_nr, exercise]
                    elif exercise=='Syncro bilateral':
                        # try:# in case the concentric phase is indeed missing
                        #     programmed_resistance = float(df_rep[((df_rep['Repetition # (per Set)']==rep) & (df_rep['Motion type']==mo_type))][col].max())#.mode())# programmed motor resistance for the concentric phase in kg
                        #     load_nr = int(pd.Series([abs(programmed_resistance-ld) for ld in side_loads[exercise]], index=side_loads[exercise].index).idxmin())
                        #     load_mass = 2*programmed_resistance
                        # except:
                        #     continue
                        programmed_resistance = float(df_rep[((df_rep['Repetition # (per Set)']==rep) & (df_rep['Motion type']==mo_type))][col].max())#.mode())# programmed motor resistance for the concentric phase in kg
                        load_nr = set_nr
                        load_mass = 2*side_loads.loc[load_nr, exercise]
        
                    # load_mass = 2*side_loads.loc[load_nr, exercise]
                    # print([programmed_resistance, load_mass])
                    if mo_type=='Con. / Resist.' and not(abs(load_mass - 2*programmed_resistance)<0.5):#and not(abs(load_mass - 2*programmed_resistance)<=0.1):
                        st.write('Warning: '+exercise+' set '+str(set_nr)+', rep '+str(rep)+'. Incorrect resistance setting ('+str(round(2*programmed_resistance, 1))+', not '+str(round(2*side_loads.loc[set_nr, exercise], 1))+')')# continue# because the resistance was set incorrectly
                    # else:
                    # display load for monitoring purposes
                    load_rep_container.text('load '+str(load_nr)+' ('+str(round(load_mass, 1))+' kg), rep '+str(rep))
    
                    # iterate through phases
                    for phase in ['out','in']:# phase='out' # phase='in'
                        # create dict to store signals within phase
                        phase_signals = {}
                        regular_phase_signals = {}
                        
                        # extract desired raw signals
                        for machine in [1,2]:# machine=1
                            phase_signals[machine] = df_rep[((df_rep['Machine index (In Syncro)']==machine) & (df_rep['Motion type']=='Con. / Resist.' if phase=='out' else df_rep['Motion type']=='Ecc. / Assist.'))][Options['use_cols']]
                            phase_signals[machine].columns = Options['use_cols']# phase_signals[machine][['Speed [m/s]', 'Acceleration [m/(s^2)]']].plot()
                                
                        if len(phase_signals[1])*len(phase_signals[2])==0:# if the phase wasn't recorded for this rep
                            st.write('rep '+str(rep)+', '+phase+': no available data')
                            continue
                    
                        # change the direction of speed to positive
                        if ((conc_direction=='in') & (phase=='in')):
                            for machine in [1,2]:# machine=1
                                for sig in ['Speed [m/s]','Acceleration [m/(s^2)]']:# sig='Speed [m/s]'
                                    phase_signals[machine][sig] = -1*phase_signals[machine][sig]
    
                        #standardize time and signals
                        for machine in [1,2]:# machine=1
                            regular_phase_signals[machine] = pd.DataFrame()
                            orig_time = phase_signals[machine]['Sample duration [s]'].cumsum()
                            regular_phase_signals[machine]['Time [s]'] = regular_time(duration=round(orig_time.max(),3))
                            for n,sig in enumerate([c for c in Options['use_cols'] if c!='Sample duration [s]']):# n,sig=0,'Position' # n,sig=1,'Speed [m/s]'
                                regular_phase_signals[machine][sig] = regular_time(duration=round(orig_time.max(),3),
                                               orig_time_and_signal = [orig_time,
                                                                      phase_signals[machine][sig]])[1]
                        # calculate F_calc 
                        if ((conc_direction=='in') & (phase=='in')):
                            if Options['F_calc_method']=='acceleration':# calculate from acceleration and assumed resistance
                                st.write('Options[F_calc_method]==acceleration not programmed yet')
        
                            if Options['F_calc_method']=='impulse':# calculate using F*dt=m*dv
                                for machine in [1,2]:# machine=1
                                    regular_phase_signals[machine]['F_calc [N]'] = F_calc_from_impulse(regular_phase_signals[machine], programmed_resistance,
                                                         side_load_mass = 0.5*load_mass)
                
                            elif Options['F_calc_method']=='energy':
                                st.write('Options[F_calc_method]==energy not programmed yet')
                
                            else:
                                st.write('no F_calc method specified')
                                        
                        else:# if phase or conc_direction is 'out', in which case F_calc is unnecessary
                            for machine in [1,2]:# machine=1
                                regular_phase_signals[machine]['F_calc [N]'] = None#np.nan
                            
                        # calculate power
                        for machine in [1,2]:# machine=1
                            regular_phase_signals[machine]['Power [W]'] = regular_phase_signals[machine]['Force [N]']*regular_phase_signals[machine]['Speed [m/s]']
                            regular_phase_signals[machine]['P_calc [W]'] = regular_phase_signals[machine]['F_calc [N]']*regular_phase_signals[machine]['Speed [m/s]']
                    
                        # calculate combined values (both sides)
                        regular_phase_signals['combined'] = pd.DataFrame(columns=regular_phase_signals[1].columns)
                        length_combined = np.min([len(regular_phase_signals[1]['Time [s]']),# sample number of machine 1
                                            len(regular_phase_signals[2]['Time [s]'])])# sample number of machine 2 (take the lower of the two)
                        regular_phase_signals['combined']['Time [s]'] = regular_phase_signals[1]['Time [s]'][:length_combined]# the shorter of the two sides
                        combined_position = pd.Series(0,index=regular_phase_signals['combined']['Time [s]'].index)
                        for i in combined_position.index[1:]:
                            dx1 = regular_phase_signals[1].loc[i,'Position [m]']-regular_phase_signals[1].loc[i-1,'Position [m]']# positional change, machine 1
                            dx2 = regular_phase_signals[2].loc[i,'Position [m]']-regular_phase_signals[2].loc[i-1,'Position [m]']# positional change, machine 2
                            dxmean = (dx1+dx2)/2.# positional change, mean of two machines
                            combined_position.loc[i] = combined_position.loc[i-1]+dxmean
                            
                        regular_phase_signals['combined']['Position [m]'] = combined_position# based on mean of two machines
                        regular_phase_signals['combined']['Speed [m/s]'] = (regular_phase_signals[1]['Speed [m/s]']+regular_phase_signals[2]['Speed [m/s]'])[:length_combined]/2# average of the two sides
                        regular_phase_signals['combined']['Acceleration [m/(s^2)]'] = (regular_phase_signals[1]['Acceleration [m/(s^2)]']+regular_phase_signals[2]['Acceleration [m/(s^2)]'])[:length_combined]/2# average of the two sides
                        regular_phase_signals['combined']['Force [N]'] = (regular_phase_signals[1]['Force [N]']+regular_phase_signals[2]['Force [N]'])[:length_combined]# sum of the two sides
                        regular_phase_signals['combined']['Power [W]'] = (regular_phase_signals[1]['Power [W]']+regular_phase_signals[2]['Power [W]'])[:length_combined]# sum of the two sides
                        regular_phase_signals['combined']['F_calc [N]'] = (regular_phase_signals[1]['F_calc [N]']+regular_phase_signals[2]['F_calc [N]'])[:length_combined]# sum of the two sides
                        regular_phase_signals['combined']['P_calc [W]'] = (regular_phase_signals[1]['P_calc [W]']+regular_phase_signals[2]['P_calc [W]'])[:length_combined]# sum of the two sides
    
                        # save phase parameters to summary table
                        i = running_rep#-1
                        Results_parameters[exercise]['summary'].loc[i,'set_nr'] = set_nr
                        Results_parameters[exercise]['summary'].loc[i,'rep_nr'] = rep
                        Results_parameters[exercise]['summary'].loc[i,'load_mass'] = load_mass
                        Results_parameters[exercise]['summary'].loc[i,'load_nr'] = load_nr
                            
                        for machine in [1,2,'combined']:# machine=1 machine='combined'
                            for par in ['Fpeak','Fmean','Fpeak_calc','Fmean_calc','vpeak','vmean','Ppeak','Pmean','Ppeak_calc','Pmean_calc','distance','duration']:# par='Fpeak'
                                Results_parameters[exercise]['summary'].loc[i,phase+'_'+('m'+str(machine) if type(machine)==int else machine)+'_'+par] = calculate_parameters(regular_phase_signals,machine)[par].values[0]
    
                        # save regular_phase_signals to summary dictionary
                        if phase==conc_direction:
                            Results_signals[exercise]['set '+str(set_nr)]['rep'+str(rep)][phase] = regular_phase_signals

        # sort the summary table
        if 'summary' in Results_parameters[exercise].keys():
            Results_parameters[exercise]['summary'].sort_values(by=['load_nr','rep_nr'])
    exercise_container.empty()
    set_container.empty()
    load_rep_container.empty()
    return Results_parameters, Results_signals, settings

def oneLiner(Results_parameters, settings, body_mass, method='mean'):# name, date, exercises=['Syncro smith bilateral', 'Syncro bilateral']
    '''
    'method' can be either 'median',  'mean' or 'best'
    method = Options['oneLiner_method']
    '''
    oneLiner = pd.DataFrame()
    oneLiner.loc[0,'name'] = settings['Name'].unique()[0]
    oneLiner.loc[0,'date'] = settings['Date'].unique()[0]
    oneLiner.loc[0,'body_mass'] = body_mass
    for exercise in Results_parameters.keys():# exercise=list(Results_parameters.keys())[0] # exercise=list(Results_parameters.keys())[list(Results_parameters.keys()).index(exercise)+1]
        try:
            df_ex_sum = Results_parameters[exercise]['summary']
        except:
            df_ex_sum = pd.DataFrame(
                columns = Results_parameters[list(Results_parameters.keys())[0]]['summary'].columns
                )
        for load_nr in range(1,7):# because there needs to be a placeholder for load 6 even if it wasn't performed # load_nr=1 # load_nr=load_nr+1
            load_nr = int(load_nr)
            df_ex_sum_ld = df_ex_sum[df_ex_sum['set_nr']==load_nr]
            for par in df_ex_sum_ld.columns[1]:# par=df_ex_sum_ld.columns[1]
                oneLiner.loc[0,'_'.join([exercise,'n_reps','load'+str(load_nr)])] = len(df_ex_sum_ld['rep_nr'])
            for par in df_ex_sum_ld.columns[2:]:# par=df_ex_sum_ld.columns[3]
                if method=='mean':
                    oneLiner.loc[0,'_'.join([exercise,par,'load'+str(load_nr)])] = df_ex_sum_ld[par].mean()
                elif method=='median':
                    oneLiner.loc[0,'_'.join([exercise,par,'load'+str(load_nr)])] = df_ex_sum_ld[par].median()
                elif method=='best':
                    oneLiner.loc[0,'_'.join([exercise,par,'load'+str(load_nr)])] = df_ex_sum_ld[par].max()
                
    oneLiner = oneLiner[list(oneLiner.columns[0:3])+sorted(list(oneLiner.columns[3:]))]
    oneLiners = {}
    for exercise in Results_parameters.keys():# exercise = 'Syncro bilateral' # exercise = 'Syncro smith bilateral' 
        oneLiners[exercise] = {}   
        cols_ex_full = [col for col in oneLiner.columns[:3]]+[col for col in oneLiner.columns[3:] if exercise in col]
        ex_full = oneLiner[cols_ex_full]# exercise, full
        oneLiners[exercise]['full'] = ex_full   
        phase = '_in_' if exercise=='Syncro smith bilateral' else '_out_'
        cols_ex_simple = [col for col in ex_full.columns[:3]]+[col for col in ex_full.columns[3:] if (((phase in col) & ('combined' in col)) or ('load_mass' in col) or ('n_reps' in col))]
        ex_simple = ex_full[cols_ex_simple]# exercise, simple
        oneLiners[exercise]['simple'] = ex_simple
        try:# if it's not the first iteration of exercise
            columns = list(list(set(list(all_ex_simple.columns)+list(ex_simple.columns))))
            general_cols = list(ex_simple.columns[:3])
            test_parameter_cols = [col for col in columns if col not in general_cols]
            columns = general_cols+sorted(test_parameter_cols)
            all_ex_simple = oneLiner[columns]
        except:# if it's the first iteration of exercise
            all_ex_simple = ex_simple

    oneLiners['all_exercises'] = {}
    oneLiners['all_exercises']['simple'] = all_ex_simple
    oneLiners['all_exercises']['full'] = oneLiner
    
    return oneLiners

def generate_excel(Results_signals, Results_parameters, oneLiners, settings, Options, **kwargs):
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
    if 'sign_digits' in kwargs:
        sign_digits = kwargs.get('sign_digits')
        for exercise in Results_parameters.keys():# exercise = list(Results_parameters.keys())[0]
            Results_parameters[exercise]['summary'] = significant_digits(Results_parameters[exercise]['summary'], sign_digits)
        for exercise in Results_signals.keys():# exercise = list(Results_signals.keys())[0] # exercise = list(Results_signals.keys())[1]
            for set_nr in Results_signals[exercise].keys():# set_nr = list(Results_signals[exercise].keys())[0]
                for rep in Results_signals[exercise][set_nr].keys():# rep = list(Results_signals[exercise][set_nr].keys())[0]
                    for phase in Results_signals[exercise][set_nr][rep].keys():# phase = list(Results_signals[exercise][set_nr][rep].keys())[0]
                        for machine in Results_signals[exercise][set_nr][rep][phase].keys():# machine = list(Results_signals[exercise][set_nr][rep][phase].keys())[0]
                            Results_signals[exercise][set_nr][rep][phase][machine] = significant_digits(Results_signals[exercise][set_nr][rep][phase][machine], sign_digits)
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
                if len(settings[((settings['Exercise']==exercise) & (settings['set_type']=='isokinetic'))])>0:
                    for s,set_nr in enumerate(settings[((settings['Exercise']==exercise) & (settings['set_type']=='isokinetic'))]['Set'], start=1):# s,set_nr = 1,settings[((settings['Exercise']==exercise) & (settings['set_type']=='isokinetic'))]['Set'].iloc[0]
                        for r,rep in enumerate(Results_signals[exercise]['set '+str(set_nr)].keys(), start=1):# r,rep = 1,list(Results_signals[exercise]['set '+str(set_nr)].keys())[0]
                            df = Results_signals[exercise]['set '+str(set_nr)][rep]['out']['combined'].copy()
                            df.insert(len(df.columns),
                                      'reversed Position [m]',
                                      df['Position [m]'] - df['Position [m]'].max())
                            df.insert(len(df.columns),
                                      'relative Position [%]',
                                      df['Position [m]'] / df['Position [m]'].max())
                            df.to_excel(writer,
                                        sheet_name='_'.join([exercise.replace(' bilateral', ''), 'isokinetic', str(s+(r*0.1))])
                                        )
                            pd.DataFrame(columns=[
                                'Name: '+settings['Name'].iloc[0],
                                'Date: '+settings['Date'].iloc[0],
                                'Exercise: '+exercise,
                                                  ]).to_excel(writer,
                                                                                                sheet_name='_'.join([exercise.replace(' bilateral', ''), 'isokinetic', str(s+(r*0.1))]),
                                                                                                index=False,
                                                                                                startcol=9,
                                                                                                startrow=0)
            if Options['all_oneLiners']:
                for exercise in oneLiners.keys():# exercise = list(oneLiners.keys())[0]
                    for entry_type in oneLiners[exercise].keys():# entry_type = list(oneLiners[exercise].keys())[0]
                        if Options['transpose_oneLiner']:
                            oneLiners[exercise][entry_type].set_index('name').T.to_excel(writer, sheet_name=exercise.replace(' bilateral', '')+'_'+entry_type)
                        else:
                            oneLiners[exercise][entry_type].set_index('name').to_excel(writer, sheet_name=exercise.replace(' bilateral', '')+'_'+entry_type)
            else:
                exercise = 'all_exercises'# exercise = list(oneLiners.keys())[0]
                entry_type = 'full'# entry_type = list(oneLiners[exercise].keys())[0]
                if Options['transpose_oneLiner']:
                    oneLiners[exercise][entry_type].set_index('name').T.to_excel(writer, sheet_name=exercise.replace(' bilateral', '')+'_'+entry_type)
                else:
                    oneLiners[exercise][entry_type].set_index('name').to_excel(writer, sheet_name=exercise.replace(' bilateral', '')+'_'+entry_type)
        
        if Options['signals_to_excel']:
            for exercise in Results_signals.keys():# exercise = list(Results_signals.keys())[0]
                for file in Results_signals[exercise].keys():# file = list(Results_signals[exercise].keys())[0]
                    for rep in Results_signals[exercise][file].keys():# rep = list(Results_signals[exercise][file].keys())[0]
                        for phase in Results_signals[exercise][file][rep].keys():# phase = list(Results_signals[exercise][file][rep].keys())[0]
                            df = Results_signals[exercise][file][rep][phase]['combined']
                            for k in [x for x in Results_signals[exercise][file][rep][phase].keys() if x!='combined']:# k=[x for x in Results_signals[exercise][file][rep][phase].keys() if x!='combined'][0]
                                df = pd.concat((df, 
                                                pd.DataFrame(
                                                    data=Results_signals[exercise][file][rep][phase][k].values,
                                                    columns=['_'.join([str(k),c]) for c in Results_signals[exercise][file][rep][phase][k].columns]
                                                    )
                                                ),
                                               axis=1)
                            df.to_excel(writer,
                                        sheet_name='_'.join([exercise.replace(' bilateral', ''), file, rep, phase]),
                                        index=False)
        writer.save()
        processed_data = output.getvalue()
        
    b64 = base64.b64encode(processed_data)
    writing_excel_container.empty()
    # return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="Results.xlsx">Download Results as Excel File</a>' # decode b'abc' => abc
    download_filename = '_'.join([settings['Name'].unique()[0],
                                  settings['Date'].unique()[0],
                                  'Results',
                                  Options['oneLiner_method']
                                  ]
                                 ) + '.xlsx'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{download_filename}">Download Results as Excel File</a>' # decode b'abc' => abc

#%%
def get_settings_from_export(data_export_file):# data_export_file = os.path.join('C:\\Users\\BASPO\\Downloads', 'RawDataExport (1).csv')
    data_export_df = pd.read_csv(data_export_file, delimiter=';')
    settings = pd.DataFrame()
    for exercise in data_export_df['Exercise'].unique():# exercise = data_export_df['Exercise'].unique()[0] # exercise = data_export_df['Exercise'].unique()[1]
        df_ex = data_export_df[data_export_df['Exercise']==exercise]
        for set_nr,s_nr in enumerate(df_ex['Set #'].unique(), start=1):# set_nr,s_nr = 1,df_ex['Set #'].unique()[0]
            df_set = df_ex[df_ex['Set #']==s_nr]
            for rep in df_set['Repetition # (per Set)'].unique():# rep = df_set['Repetition # (per Set)'].unique()[0]
                if rep==df_set['Repetition # (per Set)'].unique().min():
                    df_rep = df_set[df_set['Repetition # (per Set)']==rep]# generate appropriate keys for Results_signals and Results_parameters
                    settings = settings.append(pd.DataFrame(
                        {
                            'Name': df_rep['Client name'].unique()[0],# a unique name string
                            'Date': df_rep['Session Date & Time [ISO 8601]'].unique()[0].split('T')[0],# a unique name string
                            'Time': df_set['Set Date & Time [ISO 8601]'].min().split('T')[1].split('.')[0],
                            'Exercise': exercise,
                            'Set': set_nr,
                            'Concentric/Resisted load [kg]': df_set['Concentric/Resisted load [kg]'].max(),
                            'Eccentric/Assisted load [kg]': df_set['Eccentric/Assisted load [kg]'].max(),
                            'Concentric/Resisted speed limit [m/s]': df_set['Concentric/Resisted speed limit [m/s]'].max(),
                            'Eccentric/Assisted speed limit [m/s]': df_set['Eccentric/Assisted speed limit [m/s]'].max(),
                            'Mode': df_set['Mode'].mode(),
                            'Gear': df_set['Gear'].mode(),
                            }
                        )).reset_index(drop=True)
    # print(settings)
    return settings

def get_protocoll_data(protocol_file):#
    pf = protocol_file.read()# pf = os.path.join(os.getcwd(),'sample_data','Test_Protocol_Micah Gross_2020-10-16.xlsx')
    body_mass = np.nanmean([pd.read_excel(pf, sheet, header=1, usecols=[1], nrows=1).iloc[0,0] for sheet in ['Press', 'Pull']])
    df = pd.read_excel(pf, None, header=6, usecols=range(12), nrows=6)#.dropna(how='all').set_index('Stage', drop=False)
    df_press = df['Press'].dropna(how='all').set_index('Stage', drop=False)# pd.ExcelFile(protocol_file).parse('Press', header=6, usecols=range(12)).loc[:5].dropna(how='all').set_index('Stage', drop=False)#, nrows=6, skiprows=7
    df_pull = df['Pull'].dropna(how='all').set_index('Stage', drop=False)# pd.ExcelFile(protocol_file).parse('Pull', header=6, usecols=range(12)).loc[:5].dropna(how='all').set_index('Stage', drop=False)#, nrows=6, skiprows=7
    side_loads = pd.DataFrame({
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
    return Options

def significant_digits(x, n):
    '''
    Parameters
    ----------
    x : float
        the value or array to round.
    n : int
        the number of significant digits to which to round.

    Returns
    -------
    x_rounded : float
        the rounded value or array.
    
    https://stackoverflow.com/questions/57590046/round-a-series-to-n-number-of-significant-figures
    '''
    # xs = [123.949, 23.87, 1.9865, 0.0129500]
    # x=int(xs[0])
    # x = xs[0]
    # x = xs[1]
    # x = xs[2]
    # x = xs[3]
    # x = xs
    # x = np.array(xs)
    # x = pd.Series(xs)
    # x = pd.Series(xs, index=range(5,9))
    # x = pd.DataFrame(np.array([xs, list(reversed(xs))]).T, columns=['a','b'])
    def round_value(xi, n):
        try:# if it's a numerical value
            float(xi)
            if type(xi) == int:
                xi_rounded = xi
            if 'float' in str(type(xi)):
                offset = int(n - np.floor(np.log10(abs(xi)))) - 1
                xi_rounded = np.round(xi, offset)
                if xi_rounded==int(xi_rounded):
                    xi_rounded = int(np.round(xi, offset))
        except:# if it's not numerical
            xi_rounded = xi
        return xi_rounded
    if type(x) == int or 'float' in str(type(x)):
        x_rounded = round_value(x, n)
    
    if type(x) == list:
        x_rounded = [round_value(xi, n) for xi in x]
    if type(x) == np.ndarray:
        x_rounded = np.array(
            [round_value(xi, n) for xi in x]
            )
    if type(x) == pd.Series:
        x_rounded = pd.Series(
            [round_value(xi, n) for xi in x],
            index=x.index
            )
    if type(x) == pd.DataFrame:
        x_rounded = pd.DataFrame(
                np.array([[round_value(xi, n) for xi in x[col]] for col in x.columns]).T,
                columns=x.columns
                )
    return x_rounded
    
#%%
st.write("""

# Upper Body MLD Test

""")
st.sidebar.header('Export Reporting Options')
Options = user_input_options()
data_export_file = st.file_uploader("upload csv export file", accept_multiple_files=False)
if data_export_file is not None:
    # with open(os.path.join(os.getcwd(),'saved_variables',(data_export_file.name.split('.')[0])+'_bytesIO.txt'), 'wb') as fp:
    #     fp.write(data_export_file.getbuffer())
    settings = get_settings_from_export(data_export_file)
    with st.beta_expander('Basic info', expanded=True):
        st.write('Client name: '+settings['Name'].unique()[0])
        st.write('Test date: '+settings['Date'].unique()[0])
        st.write('Exercises: '+str(list(settings['Exercise'].unique())))
    if settings is not None:
        protocol_file = st.file_uploader('upload test protocol file (xlsx)', accept_multiple_files=False)
        if protocol_file is not None:
            # with open(os.path.join(os.getcwd(),'saved_variables','protocol_file_bytesIO.txt'), 'wb') as fp:
            #     fp.write(protocol_file.getbuffer())
            body_mass, side_loads = get_protocoll_data(protocol_file)# protocol_file = uploaded_files[uploaded_files_names.index([n for n in uploaded_files_names if 'Test_Protocol' in n][0])]
            with st.beta_expander('loads (per side)', expanded=False):
                st.write(significant_digits(side_loads, 3))
            if side_loads is not None:
                Results_parameters, Results_signals, settings = results_parameters_signals(data_export_file,
                                                                                           settings,
                                                                                           side_loads,
                                                                                           Options,
                                                                                           )
                with st.beta_expander('Settings', expanded=False):
                    st.write(significant_digits(settings, 3))
                if Results_parameters is not None:
                    oneLiners = oneLiner(Results_parameters, settings, body_mass, method=Options['oneLiner_method'])
                    if oneLiners is not None:
                        if any([Options['parameters_to_excel'], Options['signals_to_excel']]):
                            st.markdown(generate_excel(Results_signals, Results_parameters, oneLiners, settings, Options), unsafe_allow_html=True)#, sign_digits=3
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
                            # # with open(os.path.join(os.getcwd(),'saved_variables','settings_data.json'), 'w') as fp:
                            # #     json.dump(settings_data, fp)
                            # side_loads.to_json(os.path.join(os.getcwd(),'saved_variables','side_loads.json'), orient='index', date_format='iso')
                            # with open(os.path.join(os.getcwd(),'saved_variables','body_mass.json'), 'w') as fp:
                            #     json.dump(body_mass, fp)
                            # with open(os.path.join(os.getcwd(),'saved_variables','Options.json'), 'w') as fp:
                            #     json.dump(Options, fp)
                            # # # with open(os.path.join(os.getcwd(),'saved_variables','uploaded_files_names.json'), 'w') as fp:
                            # # #     json.dump([f.name for f in uploaded_files], fp)
                            # # # for f in uploaded_files:
                            # # #     with open(os.path.join(os.getcwd(),'saved_variables',(f.name.split('.')[0])+'_bytesIO.txt'), 'wb') as fp:
                            # # #         fp.write(f.getbuffer())
                            # # # for f in protocol_file:
                            # # #     # with open(os.path.join(os.getcwd(),'saved_variables',(f.name.split('.')[0])+'_bytesIO.txt'), 'wb') as fp:
                            # # #     with open(os.path.join(os.getcwd(),'saved_variables','protocol_file_bytesIO.txt'), 'wb') as fp:
                            # # #         fp.write(f.getbuffer())
                            # # # with open(os.path.join(os.getcwd(),'saved_variables','name_date_exercises.json'), 'w') as fp:
                            # # #     json.dump([name, date, exercises], fp)
                            # st.write('succeeded!')
                            st.success('done')
        
