#!/usr/bin/env python

#helper functions for cleaning time and other factors of RBR CTD, GGA, DGEU, 
#AirMar Weather Station, Optode, and Nitrate Sensors

#author: Victoria Preston
#supervisors: Anna Michel, David Nicholson
#contact: vpreston@whoi.edu, vpreston@mit.edu

import numpy as np

def calculate_julian_day(y, mo, d, h, mi, s):
    '''
    Takes a timestamp (hours in UTC) and converts to Julian Day via the appropriate formula.
    @input: year, month, day, hour, minute, seconds_total
    @output: julian day
    '''
    a = int((14 - mo)/12)
    ye = y + 4800 - a
    m = mo + 12*a - 3
    jdn = d + ((153*m + 2)/5) + 365*ye + ye/4 - ye/100 + ye/400 - 32045
    df = (h-12)/24 + mi/1440 + s/86400
    jd = float(jdn) + float(df)
    return jd

def calculate_seconds_elapsed(df_row):
    '''
    An alternative to julian day, calculates the total seconds elapsed in the month of running.
    @input: row in a dataframe which contains day, hour, minute, and second fields
    @output: second that it is in the month
    '''
    seconds_total = df_row['Day']*24*60*60 + df_row['Hour']*60*60 + df_row['Minute']*60 + df_row['Second']
    return seconds_total

def seconds_elapsed(df):
    '''
    Adds a seconds elapsed column to the dataframe
    @input: dataframe
    @output: modifies dataframe without explicit output
    '''
    df.loc[:,'Seconds_Elapsed'] = df.apply(lambda row: calculate_seconds_elapsed(row), axis=1)
    return

def global_time_column(df):
    '''
    Creates a global time column based upon the initial timestamp of data collection
    @input: dataframe
    @output: none; dataframe rewritten with Julian_Data column
    '''
    df.loc[:,'Julian_Date'] = df.apply(lambda row: calculate_julian_day(row['Year'], 
                                                                        row['Month'], 
                                                                        row['Day'], 
                                                                        float(row['Hour']), 
                                                                        float(row['Minute']), 
                                                                        float(row['Second'])),axis=1)
    return

def handle_ctd_time(data):
    '''
    Handle seperating the timestamp object into individual data columns
    @input: dataframe
    @output: none; dataframe rewritten in processing
    '''
    data.loc[:,'Year'] = data['Time'].str.split(' ').str.get(0).str.split('-').str.get(0).astype('int')
    data.loc[:,'Month'] = data['Time'].str.split(' ').str.get(0).str.split('-').str.get(1).astype('int')
    data.loc[:,'Day'] = data['Time'].str.split(' ').str.get(0).str.split('-').str.get(2).astype('int')
    data.loc[:,'Hour'] = data['Time'].str.split(' ').str.get(1).str.split(':').str.get(0).astype('float')
    data.loc[:,'Minute'] = data['Time'].str.split(' ').str.get(1).str.split(':').str.get(1).astype('float')
    data.loc[:,'Second'] = data['Time'].str.split(' ').str.get(1).str.split(':').str.get(2).astype('float')
    global_time_column(data)
    seconds_elapsed(data)
    return

def handle_gga_time(data):
    '''
    Handle seperating the timestamp object into individual data columns
    @input: dataframe
    @output: none; dataframe rewritten in processing
    '''
    data.loc[:,'Year'] = data['Time'].str.split(' ').str.get(0).str.split('/').str.get(2).astype('int')
    data.loc[:,'Month'] = data['Time'].str.split(' ').str.get(0).str.split('/').str.get(0).astype('int')
    data.loc[:,'Day'] = data['Time'].str.split(' ').str.get(0).str.split('/').str.get(1).astype('int')
    data.loc[:,'Hour'] = data['Time'].str.split(' ').str.get(1).str.split(':').str.get(0).astype('float')
    data.loc[:,'Minute'] = data['Time'].str.split(' ').str.get(1).str.split(':').str.get(1).astype('float')
    data.loc[:,'Second'] = data['Time'].str.split(' ').str.get(1).str.split(':').str.get(2).astype('float')
    global_time_column(data)
    seconds_elapsed(data)
    return

def handle_airmar_time(data):
    '''
    Handle seperating the timestamp object into individual data columns
    @input: dataframe
    @output: none; dataframe rewritten in processing
    '''
    data.loc[:,'Year'] = data.apply(lambda x : int(x['year']),axis=1)
    data.loc[:, 'Month'] = data.apply(lambda x : int(x['month']),axis=1)
    data.loc[:, 'Day'] = data.apply(lambda x : int(x['day']),axis=1)
    data.loc[:,'Hour'] = data.apply(lambda x : float(str(x['TOD'])[0:2]),axis=1)
    data.loc[:,'Minute'] = data.apply(lambda x : float(str(x['TOD'])[2:4]),axis=1)
    data.loc[:,'Second'] = data.apply(lambda x : float(str(x['TOD'])[4:]),axis=1)
    global_time_column(data)
    seconds_elapsed(data)
    return

def handle_dge_time(data):
    '''
    Handle seperating the timestamp object into individual data columns
    @input: dataframe
    @output: none; dataframe rewritten in processing
    '''
    data.loc[:,'Year'] = data['TIMESTAMP'].str.split(' ').str.get(0).str.split('-').str.get(0).astype('int')
    data.loc[:,'Month'] = data['TIMESTAMP'].str.split(' ').str.get(0).str.split('-').str.get(1).astype('int')
    data.loc[:,'Day'] = data['TIMESTAMP'].str.split(' ').str.get(0).str.split('-').str.get(2).astype('int')
    data.loc[:,'Hour'] = data['TIMESTAMP'].str.split(' ').str.get(1).str.split(':').str.get(0).astype('float' )
    data.loc[:,'Minute'] = data['TIMESTAMP'].str.split(' ').str.get(1).str.split(':').str.get(1).astype('float')
    data.loc[:,'Second'] = data['TIMESTAMP'].str.split(' ').str.get(1).str.split(':').str.get(2).astype('float')
    global_time_column(data)
    seconds_elapsed(data)
    return

def handle_nitrate_time(data):
    '''
    Handle seperating the timestamp object into individual data columns
    @input: dataframe
    @output: none; dataframe rewritten in processing
    '''
    data.loc[:,'Year'] = data.apply(lambda x : int(str(x['2017192'])[0:4]),axis=1)
    data.loc[:,'Month'] = data.apply(lambda x : int(str(x['2017192'])[4:])/30+1,axis=1)
    data.loc[:,'Day'] = data.apply(lambda x : int(str(x['2017192'])[4:])%30-1,axis=1)
    data.loc[:,'Hour'] = data.apply(lambda x : int(x['20.364744']),axis=1)
    data.loc[:,'Minute'] = data.apply(lambda x : int((float(x['20.364744']) - float(x['Hour']))  * 60),axis=1)
    data.loc[:,'Second'] = data.apply(lambda x : float(((float(x['20.364744']) - float(x['Hour'])) * 60 - float(x['Minute'])) * 60),axis=1)
    
    data.loc[:,'Hour'] = data.apply(lambda x : int(x['20.364744']),axis=1)
    
    global_time_column(data)
    seconds_elapsed(data)
    return

def dms2dd(info):
    degrees = float(str(info)[0:2])
    minutes = float(str(info)[2:]) / 60.0
    dd = degrees + minutes
    return dd

def get_distance(info):
    plume_center = [np.radians(-70.684092), np.radians(41.758048)]
    info_point = [np.radians(info['lon_mod']), np.radians(info['lat_mod'])]
    R=6373.0


    dlon = info_point[0] - plume_center[0]
    dlat = info_point[1] - plume_center[1]

    a = np.sin(dlat / 2)**2 + np.cos(info_point[1]) * np.cos(plume_center[1]) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c* 1000.0 * np.sign(info_point[0] - plume_center[0])

def get_side(info):
    return (1.0 if info['distance'] < 0.0 else 0.0)

def get_window(info):
    return (1.0 if np.fabs(info['distance']) < 200.0 else 0.0)

def convert_gas(gr, gas):
    if gas == "CH4":
        eff = 0.1511
        gppm = 1.834
    elif gas == "CO2":
        eff = 0.806
        gppm = 0.0
    peq = 235.7
    
    ui = peq*gppm/1000.
    ua = gr*peq/1000.
    
    return (ua - ui)/eff + ui

def clean_ctd(data, min_time=None, max_time=None):
    '''
    Wrapper function for processing all of the CTD dataframe changes
    @input: dataframe to edit, bounds of time for dropping values
    @output: writes to dataframe
    '''
    handle_ctd_time(data)
    if min_time != None:
        data = data.drop(data[data.Julian_Date < min_time].index)
    if max_time != None:
        data = data.drop(data[data.Julian_Date > max_time].index)
    return data

def clean_gga(data, min_time=None, max_time=None):
    '''
    Wrapper function for processing all of the GGA dataframe changes
    @input: dataframe to edit, bounds for dropping values
    @output: writes to dataframe
    '''
    data = data.dropna()
    data = data.drop(data[data.Time == '-----BEGIN'].index)
    handle_gga_time(data)

    data.loc[:,'CH4_ppm'] = data.apply(lambda x : x['     [CH4]_ppm'], axis=1)
    data.loc[:,'CO2_ppm'] = data.apply(lambda x : x['     [CO2]_ppm'], axis=1)

    data = data.drop(['     [CH4]_ppm', '     [CO2]_ppm'], axis=1)

    data.loc[:,'CH4_ppm_adjusted'] = data.apply(lambda x: convert_gas(x['CH4_ppm'], 'CH4'), axis=1)
    data.loc[:,'CO2_ppm_adjusted'] = data.apply(lambda x: convert_gas(x['CO2_ppm'], 'CO2'), axis=1)
    
    if min_time != None:
        data = data.drop(data[data.Julian_Date < min_time].index)
    if max_time != None:
        data = data.drop(data[data.Julian_Date > max_time].index)
    
    return data

def clean_optode(data, min_time=None, max_time=None):
    '''
    Wrapper function for processing all of the optode dataframe changes
    @input: dataframe to edit, bounds for dropping values
    @output: writes to dataframe
    '''
    data.loc[:,'Julian_Date'] = data.apply(lambda x : x.posixtime / 86400.0 + 2440587.5 - 0.0416600001, axis=1)
    if min_time != None:
        data = data.drop(data[data.Julian_Date < min_time].index)
    if max_time != None:
        data = data.drop(data[data.Julian_Date > max_time].index)
    return data

def clean_airmar(data, min_time=None, max_time=None):
    '''
    Wrapper function for processing all of the airmar dataframe changes
    @input: dataframe to edit, bounds for dropping values
    @output: writes to dataframe
    '''
    data = data.dropna()
    handle_airmar_time(data)

    data.loc[:, 'lon_mod'] = data.apply(lambda x : -dms2dd(x.lon), axis=1)
    data.loc[:, 'lat_mod'] = data.apply(lambda x : dms2dd(x.lat), axis=1)
    data.loc[:, 'distance'] = data.apply(lambda x: get_distance(x), axis=1)
    data.loc[:, 'side'] = data.apply(lambda x: get_side(x), axis=1)
    data.loc[:, 'in_range'] = data.apply(lambda x: get_window(x), axis=1)

    if min_time != None:
        data = data.drop(data[data.Julian_Date < min_time].index)
    if max_time != None:
        data = data.drop(data[data.Julian_Date > max_time].index)
    return data

def clean_dgeu(data, min_time=None, max_time=None):
    '''
    Wrapper function for processing all of the dgeu dataframe changes
    @input: dataframe to edit, bounds for dropping values
    @output: writes to dataframe
    '''
    data = data.dropna()
    data = data.drop(data[data.TIMESTAMP == 'TS'].index)
    handle_dge_time(data)
    if min_time != None:
        data = data.drop(data[data.Julian_Date < min_time].index)
    if max_time != None:
        data = data.drop(data[data.Julian_Date > max_time].index)
    return data

def clean_nitrate(data, min_time=None, max_time=None):
    '''
    Wrapper function for processing all of the nitrate dataframe changes
    @input: dataframe to edit, bounds for dropping values
    @output: writes to dataframe
    '''
    data = data.drop(data[data.SATSDF0342 == 'SATSDF0342'].index)
    handle_nitrate_time(data)
    data = data.drop(data[data['0.00'] < 0].index)

    for i in range(3):
        mean = data['0.00'].mean()
        std = data['0.00'].std()
        data = data.drop(data[np.fabs(data['0.00'] - mean) > 2*std].index)

    if min_time != None:
        data = data.drop(data[data.Julian_Date < min_time].index)
    if max_time != None:
        data = data.drop(data[data.Julian_Date > max_time].index)
    return data
