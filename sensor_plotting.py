#!/usr/bin/env python

#helper functions for plotting data from PANDAS data frames for sensor information

#author: Victoria Preston
#supervisors: Anna Michel, David Nicholson
#contact: vpreston@whoi.edu, vpreston@mit.edu

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.image import NonUniformImage
from matplotlib import cm
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import seaborn as sns
from descartes import PolygonPatch
from shapely.geometry import LineString
import shapefile
from scipy.optimize import curve_fit

######
#Helper Class
######
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

######
#Temporal Graphs
######

def plot_timequad(plt, x, y1, y2, y3, y4, xlabel, y1label, y2label, y3label, y4label, title):
    plt.suptitle(title, fontsize=18)

    plt.subplot(221)
    plt.plot(x, y1)
    plt.title(y1label)
    plt.grid(True)

    plt.subplot(222)
    plt.plot(x, y2)
    plt.title(y2label)
    plt.grid(True)

    plt.subplot(223)
    plt.plot(x, y3)
    plt.title(y3label)
    plt.grid(True)

    plt.subplot(224)
    plt.plot(x, y4)
    plt.title(y4label)
    plt.grid(True)

    plt.subplots_adjust(top=0.8, bottom=0.12, left=0.10, right=0.95, hspace=0.70,
                        wspace=0.5)

def plot_multiaxes(host, plt, x, data, labels, title):
    plt.suptitle(title, fontsize=18)
    plt.subplots_adjust(right=0.75)

    axes_to_make = len(data)
    offset = 60
    axis = [host]
    for i in range(axes_to_make - 1):
        axis.append(host.twinx())

    for axi in axis[2:]:
        new_fixed_axis = axi.get_grid_helper().new_fixed_axis
        axi.axis["right"] = new_fixed_axis(loc="right", axes=axi,
                                            offset=(offset, 0))
        offset = offset + 60
        axi.axis["right"].toggle(all=True)

    host.set_xlabel(labels[0])
    for axi, label in zip(axis, labels[1:]):
        axi.set_ylabel(label)

    plots = []
    for axi, datum, label in zip(axis, data, labels[1:]):
        plots.append(axi.plot(x, datum, label=label)[0])

    host.legend()

    host.axis["left"].label.set_color(plots[0].get_color())
    for axi, p in zip(axis[1:], plots[1:]):
        axi.axis["right"].label.set_color(p.get_color())

def plot_norm(host, plt, x, data, labels, title):
    plt.suptitle(title, fontsize=18)

    host.set_xlabel("Time, Julian Date")
    host.set_ylabel("Normalized Measurement")

    plots = []
    for datum, label in zip(data, labels):
        plots.append(host.plot(x, datum, label=label)[0])

    host.legend()

def plot_transectsuite(ta, transect_name):
    x = ta['Unnamed: 0_level_0']['Unnamed: 0_level_1'][1:]

    # Plot instruments
    plt.figure(1)
    plot_timequad(plt,
                     x,
                     ta['ctd']['Salinity'][1:],
                     ta['ctd']['Temperature'][1:],
                     ta['ctd']['Pressure'][1:],
                     ta['ctd']['Conductivity'][1:],
                     'Time (Julian Date)',
                     'Salinity',
                     'Temperature',
                     'Pressure',
                     'Conductivity',
                     ('CTD Sensor Readings: ' + transect_name))
    plt.figure(2)
    plot_timequad(plt,
                     x, 
                     ta['gga']['CH4_ppm'][1:], 
                     ta['gga']['CO2_ppm'][1:],
                     ta['gga']['CH4_ppm_adjusted'][1:],
                     ta['gga']['CO2_ppm_adjusted'][1:],
                     'Time (Julian_Date)',
                     'Raw Methane Reading',
                     'Raw Carbon Dioxide Reading',
                     'Adjusted Methane Reading',
                     'Adjusted Carbon Dioxide Reading',
                     ('GGA Sensor Readings: ' + transect_name))

    plt.figure(3)
    plot_timequad(plt,
                     x, 
                     ta['op']['O2Concentration'][1:], 
                     ta['op']['CalPhase'][1:],
                     ta['op']['AirSaturation'][1:],
                     ta['op']['Temperature'][1:],
                     'Time (Julian_Date)',
                     'Oxygen Concentration',
                     'CalPhase',
                     'Air Saturation',
                     'Temperature',
                     ('Optode Sensor Readings: ' + transect_name))

    plt.figure(4)
    plot_timequad(plt,
                     x, 
                     ta['airmar']['COG_T'][1:], 
                     ta['airmar']['SOG_K'][1:],
                     ta['airmar']['wind_dir_T'][1:],
                     ta['airmar']['wind_speed_M'][1:],
                     'Time (Julian_Date)',
                     'COG_T',
                     'SOG_K',
                     'Wind Direction',
                     'Wind Speed',
                     ('Weather Station Readings: ' + transect_name))

    plt.figure(6)
    plt.suptitle(('Nitrate Reading: ' + transect_name), fontsize=18)

    plt.plot(x, ta['nitrate']['0.00'][1:])
    plt.grid(True)

    plt.show()

    # Plot the coordinates over time
    plt.figure(1)
    plt.plot(ta['airmar']['lon_mod'][1:], ta['airmar']['lat_mod'][1:])
    plt.suptitle(('Outline Of ' + transect_name), fontsize=18)

    polys  = shapefile.Reader('./GIS_Data/River.shp')
    poly = polys.iterShapes().next().__geo_interface__
    ax = plt.gca()
    ax.add_patch(PolygonPatch(poly, fc="#6699cc", ec="#6699cc", alpha=0.5, zorder=2))

    plt.show()

    # Multiaxes
    host = host_subplot(111, axes_class=AA.Axes)
    plt.figure(1)

    data = [ta['gga']['CH4_ppm_adjusted'][1:],
            ta['gga']['CO2_ppm_adjusted'][1:],
            ta['nitrate']['0.00'][1:],
            ta['op']['O2Concentration'][1:],
            ta['ctd']['Salinity'][1:]]

    labels = ['Time(Julian_Date)', 'Methane', 'CO2', 'Nitrate', 'Oxygen', 'Salinity']

    plot_multiaxes(host, plt, x, data, labels, ('Instrument Readings: ' + transect_name))

    plt.draw()
    plt.show()

    # Normalized
    host2 = host_subplot(111, axes_class=AA.Axes)
    plt.figure(1)

    meth_norm = (ta['gga']['CH4_ppm_adjusted'] - ta['gga']['CH4_ppm_adjusted'].mean())/ta['gga']['CH4_ppm_adjusted'].std()
    co2_norm = (ta['gga']['CO2_ppm_adjusted'] - ta['gga']['CO2_ppm_adjusted'].mean())/ta['gga']['CO2_ppm_adjusted'].std()
    nit_norm = (ta['nitrate']['0.00'] - ta['nitrate']['0.00'].mean())/ta['nitrate']['0.00'].std()
    oxy_norm = (ta['op']['O2Concentration'] - ta['op']['O2Concentration'].mean())/ta['op']['O2Concentration'].std()
    sal_norm = (ta['ctd']['Salinity'] - ta['ctd']['Salinity'].mean())/ta['ctd']['Salinity'].std()

    data = [meth_norm[1:], co2_norm[1:], nit_norm[1:], oxy_norm[1:], sal_norm[1:]]
    labels = ['Methane', 'CO2', 'Nitrate', 'Oxygen', 'Salinity']
    plot_norm(host2, plt, x, data, labels, ('Normalized Instrument Readings: ' + transect_name))

    plt.draw()
    plt.show()

    # Reduced Dimensionality
    host3 = host_subplot(111, axes_class=AA.Axes)
    plt.figure(1)

    data = [meth_norm[1:], co2_norm[1:], sal_norm[1:]]
    labels = ['Methane', 'CO2', 'Salinity']
    plot_norm(host3, plt, x, data, labels, ('Normalized Instrument Readings: ' + transect_name))

    plt.draw()
    plt.show()

######
#Property Graphs
######

def plot_pvp(x, y, labels, title, fit=None):
    plt.figure(1)
    plt.plot(x, y, 'o')
    if fit:
        n = np.polyfit(x, y, fit)
        plt.plot(np.unique(x), np.poly1d(n)(np.unique(x)))
        print n
    plt.suptitle(title, fontsize=18)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    plt.show()

def plot_pvp_distance(x, y, c, labels, title):
    cmap = cm.bwr
    plt.figure(1)
    norm = MidpointNormalize(midpoint=0)
    plt.scatter(x, y, c=c, cmap=cmap, norm=norm, alpha=0.5, edgecolors='face')
    plt.suptitle(title, fontsize=18)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    cb = plt.colorbar()
    cb.set_label('Distance (m)')
    plt.show()

def plot_pvp_inrange(x,y,c,labels,title):
    kcolors = ['red' if cat==1 else 'blue' for cat in c]
    plt.figure(1)
    plt.scatter(x, y, c=kcolors, alpha=0.3, edgecolors='face')
    plt.suptitle(title, fontsize=18)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.show()

def plot_pvp_transect(ta, tb, tc, td, te, instruments, elements, labels, title):
    x = ta[instruments[0]][elements[0]][1:]
    y = ta[instruments[1]][elements[1]][1:]

    x1 = tb[instruments[0]][elements[0]][1:]
    y1 = tb[instruments[1]][elements[1]][1:]

    x2 = tc[instruments[0]][elements[0]][1:]
    y2 = tc[instruments[1]][elements[1]][1:]

    x3 = td[instruments[0]][elements[0]][1:]
    y3 = td[instruments[1]][elements[1]][1:]

    x4 = te[instruments[0]][elements[0]][1:]
    y4 = te[instruments[1]][elements[1]][1:]

    plt.plot(x, y, 'bo', label='TransectA', alpha=0.5)
    plt.plot(x1, y1, 'ro', label='TransectB', alpha=0.5)
    plt.plot(x2, y2, 'go', label='TransectC', alpha=0.5)
    plt.plot(x3, y3, 'ko', label='TransectD', alpha=0.5)
    plt.plot(x4, y4, 'yo', label='TransectE', alpha=0.5)


    plt.title(title)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend()
    plt.show()

def plot_pvptransect(df, transect_name):
    plot_pvp(df['ctd']['Salinity'], df['ctd']['Temperature'], ['Salinity', 'Temperature'], ('CTD: Salinity versus Temperature - ' + transect_name))
    plot_pvp(df['gga']['CH4_ppm_adjusted'], df['gga']['CO2_ppm_adjusted'], ['Methane', 'CO2'], ('GGA: Methane vs Carbon Dioxide - ' + transect_name))
    plot_pvp(df['op']['O2Concentration'], df['op']['Temperature'], ['O2', 'Temperature'], ('Optode: O2 Concentration versus Temperature - ' + transect_name))

    plot_pvp(df['ctd']['Salinity'], df['gga']['CH4_ppm_adjusted'], ['Salinity', 'Methane'], ('Salinity v Methane - ' + transect_name))
    plot_pvp(df['ctd']['Salinity'], df['gga']['CO2_ppm_adjusted'], ['Salinity', 'CO2'], ('Salinity v Carbon Dioxide - ' + transect_name))
    plot_pvp(df['ctd']['Salinity'], df['op']['O2Concentration'], ['Salinity', 'O2'], ('Salinity v Oxygen Concentration - ' + transect_name))
    plot_pvp(df['ctd']['Salinity'], df['nitrate']['0.00'], ['Salinity', 'Nitrate'], ('Salinity v Nitrate - ' + transect_name))

    plot_pvp(df['ctd']['Temperature'], df['op']['Temperature'], ['CTD Temp', 'Optode Temp'], ('Comparing Temperature Measures - ' + transect_name))
    plot_pvp(df['ctd']['Temperature'], df['gga']['CH4_ppm_adjusted'], ['Temperature', 'Methane'], ('Temperature v Methane - ' + transect_name))
    plot_pvp(df['ctd']['Temperature'], df['gga']['CO2_ppm_adjusted'], ['Temperature', 'CO2'], ('Temperature v Carbon Dioxide - ' + transect_name))
    plot_pvp(df['ctd']['Temperature'], df['op']['O2Concentration'], ['Temperature', 'O2'], ('Temperature v Oxygen Concentration - ' + transect_name))
    plot_pvp(df['ctd']['Temperature'], df['nitrate']['0.00'], ['Temperature', 'Nitrate'], ('Temperature v Nitrate - ' + transect_name))

    plot_pvp(df['gga']['CH4_ppm_adjusted'], df['op']['O2Concentration'], ['Methane', 'O2'], ('Methane v Oxygen Concentration - ' + transect_name))
    plot_pvp(df['gga']['CH4_ppm_adjusted'], df['nitrate']['0.00'], ['Methane', 'Nitrate'], ('Methane v Nitrate - ' + transect_name))

    plot_pvp(df['gga']['CO2_ppm_adjusted'], df['op']['O2Concentration'], ['CO2', 'O2'], ('CO2 v Oxygen Concentration - ' + transect_name))
    plot_pvp(df['gga']['CO2_ppm_adjusted'], df['nitrate']['0.00'], ['CO2', 'Nitrate'], ('CO2 v Nitrate - ' + transect_name))

    plot_pvp(df['nitrate']['0.00'], df['op']['O2Concentration'], ['Nitrate', 'O2'], ('Nitrate v Oxygen Concentration - ' + transect_name))

def plot_pvpseries(df):
    plot_pvp_distance(df['ctd']['Salinity'], df['ctd']['Temperature'], df['airmar']['distance'], ['Salinity', 'Temperature'], ('CTD: Salinity versus Temperature'))
    plot_pvp_distance(df['gga']['CH4_ppm_adjusted'], df['gga']['CO2_ppm_adjusted'], df['airmar']['distance'], ['Methane', 'CO2'], ('GGA: Methane vs Carbon Dioxide'))
    plot_pvp_distance(df['op']['O2Concentration'], df['op']['Temperature'], df['airmar']['distance'], ['O2', 'Temperature'], ('Optode: O2 Concentration versus Temperature'))

    plot_pvp_distance(df['ctd']['Salinity'], df['gga']['CH4_ppm_adjusted'], df['airmar']['distance'], ['Salinity', 'Methane'], ('Salinity v Methane'))
    plot_pvp_distance(df['ctd']['Salinity'], df['gga']['CO2_ppm_adjusted'], df['airmar']['distance'], ['Salinity', 'CO2'], ('Salinity v Carbon Dioxide'))
    plot_pvp_distance(df['ctd']['Salinity'], df['op']['O2Concentration'], df['airmar']['distance'], ['Salinity', 'O2'], ('Salinity v Oxygen Concentration'))
    plot_pvp_distance(df['ctd']['Salinity'], df['nitrate']['0.00'], df['airmar']['distance'], ['Salinity', 'Nitrate'], ('Salinity v Nitrate'))

    plot_pvp_distance(df['ctd']['Temperature'], df['op']['Temperature'], df['airmar']['distance'], ['CTD Temp', 'Optode Temp'], ('Comparing Temperature Measures'))
    plot_pvp_distance(df['ctd']['Temperature'], df['gga']['CH4_ppm_adjusted'], df['airmar']['distance'], ['Temperature', 'Methane'], ('Temperature v Methane'))
    plot_pvp_distance(df['ctd']['Temperature'], df['gga']['CO2_ppm_adjusted'], df['airmar']['distance'], ['Temperature', 'CO2'], ('Temperature v Carbon Dioxide'))
    plot_pvp_distance(df['ctd']['Temperature'], df['op']['O2Concentration'], df['airmar']['distance'], ['Temperature', 'O2'], ('Temperature v Oxygen Concentration'))
    plot_pvp_distance(df['ctd']['Temperature'], df['nitrate']['0.00'], df['airmar']['distance'], ['Temperature', 'Nitrate'], ('Temperature v Nitrate'))

    plot_pvp_distance(df['gga']['CH4_ppm_adjusted'], df['op']['O2Concentration'], df['airmar']['distance'], ['Methane', 'O2'], ('Methane v Oxygen Concentration'))
    plot_pvp_distance(df['gga']['CH4_ppm_adjusted'], df['nitrate']['0.00'], df['airmar']['distance'], ['Methane', 'Nitrate'], ('Methane v Nitrate'))

    plot_pvp_distance(df['gga']['CO2_ppm_adjusted'], df['op']['O2Concentration'], df['airmar']['distance'], ['CO2', 'O2'], ('CO2 v Oxygen Concentration'))
    plot_pvp_distance(df['gga']['CO2_ppm_adjusted'], df['nitrate']['0.00'], df['airmar']['distance'], ['CO2', 'Nitrate'], ('CO2 v Nitrate'))

    plot_pvp_distance(df['nitrate']['0.00'], df['op']['O2Concentration'], df['airmar']['distance'], ['Salinity', 'O2'], ('Nitrate v Oxygen Concentration'))

def plot_pvpseriestransect(ta, tb, tc, td, te):
    plot_pvp_transect(ta, tb, tc, td, te, ['ctd', 'ctd'], ['Salinity', 'Temperature'], ['Salinity', 'Temperature'], ('CTD: Salinity versus Temperature'))
    plot_pvp_transect(ta, tb, tc, td, te, ['gga', 'gga'], ['CH4_ppm_adjusted', 'CO2_ppm_adjusted'], ['Methane', 'CO2'], ('GGA: Methane vs Carbon Dioxide'))
    plot_pvp_transect(ta, tb, tc, td, te, ['op', 'op'], ['O2Concentration','Temperature'], ['O2', 'Temperature'], ('Optode: O2 Concentration versus Temperature'))

    plot_pvp_transect(ta, tb, tc, td, te, ['ctd', 'gga'],['Salinity', 'CH4_ppm_adjusted'],['Salinity', 'Methane'], ('Salinity v Methane'))
    plot_pvp_transect(ta, tb, tc, td, te, ['ctd', 'gga'],['Salinity', 'CO2_ppm_adjusted'],['Salinity', 'CO2'], ('Salinity v Carbon Dioxide'))
    plot_pvp_transect(ta, tb, tc, td, te, ['ctd', 'op'],['Salinity','O2Concentration'], ['Salinity', 'O2'], ('Salinity v Oxygen Concentration'))
    plot_pvp_transect(ta, tb, tc, td, te, ['ctd', 'nitrate'],['Salinity','0.00'], ['Salinity', 'Nitrate'], ('Salinity v Nitrate'))

    plot_pvp_transect(ta, tb, tc, td, te, ['ctd','op'], ['Temperature', 'Temperature'], ['CTD Temp', 'Optode Temp'], ('Comparing Temperature Measures'))
    plot_pvp_transect(ta, tb, tc, td, te, ['ctd', 'gga'], ['Temperature', 'CH4_ppm_adjusted'], ['Temperature', 'Methane'], ('Temperature v Methane'))
    plot_pvp_transect(ta, tb, tc, td, te, ['ctd', 'gga'], ['Temperature', 'CO2_ppm_adjusted'], ['Temperature', 'CO2'], ('Temperature v Carbon Dioxide'))
    plot_pvp_transect(ta, tb, tc, td, te, ['ctd','op'], ['Temperature', 'O2Concentration'], ['Temperature', 'O2'], ('Temperature v Oxygen Concentration'))
    plot_pvp_transect(ta, tb, tc, td, te, ['ctd', 'nitrate'], ['Temperature', '0.00'], ['Temperature', 'Nitrate'], ('Temperature v Nitrate'))

    plot_pvp_transect(ta, tb, tc, td, te, ['gga', 'op'], ['CH4_ppm_adjusted','O2Concentration'], ['Methane', 'O2'], ('Methane v Oxygen Concentration'))
    plot_pvp_transect(ta, tb, tc, td, te, ['gga', 'nitrate'], ['CH4_ppm_adjusted', '0.00'], ['Methane', 'Nitrate'], ('Methane v Nitrate'))

    plot_pvp_transect(ta, tb, tc, td, te, ['gga', 'op'], ['CO2_ppm_adjusted','O2Concentration'], ['CO2', 'O2'], ('CO2 v Oxygen Concentration'))
    plot_pvp_transect(ta, tb, tc, td, te, ['gga', 'nitrate'], ['CO2_ppm_adjusted', '0.00'], ['CO2', 'Nitrate'], ('CO2 v Nitrate'))

    plot_pvp_transect(ta, tb, tc, td, te, ['nitrate', 'op'], ['0.00', 'O2Concentration'], ['Nitrate', 'O2'], ('Nitrate v Oxygen Concentration'))

def plot_pvprange(df):
    plot_pvp_inrange(df['ctd']['Salinity'], df['ctd']['Temperature'], df['airmar']['in_range'], ['Salinity', 'Temperature'], ('CTD: Salinity versus Temperature'))
    plot_pvp_inrange(df['gga']['CH4_ppm_adjusted'], df['gga']['CO2_ppm_adjusted'], df['airmar']['in_range'], ['Methane', 'CO2'], ('GGA: Methane vs Carbon Dioxide'))
    plot_pvp_inrange(df['op']['O2Concentration'], df['op']['Temperature'], df['airmar']['in_range'], ['O2', 'Temperature'], ('Optode: O2 Concentration versus Temperature'))

    plot_pvp_inrange(df['ctd']['Salinity'], df['gga']['CH4_ppm_adjusted'], df['airmar']['in_range'], ['Salinity', 'Methane'], ('Salinity v Methane'))
    plot_pvp_inrange(df['ctd']['Salinity'], df['gga']['CO2_ppm_adjusted'], df['airmar']['in_range'], ['Salinity', 'CO2'], ('Salinity v Carbon Dioxide'))
    plot_pvp_inrange(df['ctd']['Salinity'], df['op']['O2Concentration'], df['airmar']['in_range'], ['Salinity', 'O2'], ('Salinity v Oxygen Concentration'))
    plot_pvp_inrange(df['ctd']['Salinity'], df['nitrate']['0.00'], df['airmar']['in_range'], ['Salinity', 'Nitrate'], ('Salinity v Nitrate'))

    plot_pvp_inrange(df['ctd']['Temperature'], df['op']['Temperature'], df['airmar']['in_range'], ['CTD Temp', 'Optode Temp'], ('Comparing Temperature Measures'))
    plot_pvp_inrange(df['ctd']['Temperature'], df['gga']['CH4_ppm_adjusted'], df['airmar']['in_range'], ['Temperature', 'Methane'], ('Temperature v Methane'))
    plot_pvp_inrange(df['ctd']['Temperature'], df['gga']['CO2_ppm_adjusted'], df['airmar']['in_range'], ['Temperature', 'CO2'], ('Temperature v Carbon Dioxide'))
    plot_pvp_inrange(df['ctd']['Temperature'], df['op']['O2Concentration'], df['airmar']['in_range'], ['Temperature', 'O2'], ('Temperature v Oxygen Concentration'))
    plot_pvp_inrange(df['ctd']['Temperature'], df['nitrate']['0.00'], df['airmar']['in_range'], ['Temperature', 'Nitrate'], ('Temperature v Nitrate'))

    plot_pvp_inrange(df['gga']['CH4_ppm_adjusted'], df['op']['O2Concentration'], df['airmar']['in_range'], ['Methane', 'O2'], ('Methane v Oxygen Concentration'))
    plot_pvp_inrange(df['gga']['CH4_ppm_adjusted'], df['nitrate']['0.00'], df['airmar']['in_range'], ['Methane', 'Nitrate'], ('Methane v Nitrate'))

    plot_pvp_inrange(df['gga']['CO2_ppm_adjusted'], df['op']['O2Concentration'], df['airmar']['in_range'], ['CO2', 'O2'], ('CO2 v Oxygen Concentration'))
    plot_pvp_inrange(df['gga']['CO2_ppm_adjusted'], df['nitrate']['0.00'], df['airmar']['in_range'], ['CO2', 'Nitrate'], ('CO2 v Nitrate'))

    plot_pvp_inrange(df['nitrate']['0.00'], df['op']['O2Concentration'], df['airmar']['in_range'], ['Salinity', 'O2'], ('Nitrate v Oxygen Concentration'))

def plot_transect_pvpdistance_master(df, ta, tb, tc, td, te):
    t = [ta, tb, tc, td, te]
    labels = ['A', 'B', 'C', 'D', 'E']
    cmap = cm.bwr
    norm = MidpointNormalize(midpoint=0)
    cmin = df['airmar']['distance'].min()
    cmax = df['airmar']['distance'].max()
    
    #Salinity vs Temperature
    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('CTD: Salinity versus Temperature', fontsize=18, y=1.08)
    plt.figure(1)
    axs = np.array(axs)
    for i in range(len(axs.reshape(-1))):
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['ctd']['Salinity'], 
                                t[i]['ctd']['Temperature'], 
                                c=t[i]['airmar']['distance'],
                                norm=norm, 
                                s=5, 
                                alpha=0.5, 
                                lw=0, 
                                cmap=cmap,
                                vmin=cmin,
                                vmax=cmax)
        axs[i].set_xlabel('Salinity')
        axs[i].set_ylabel('Temperature')

        # for label in (axs[i].get_xticklabels() + axs[i].get_yticklabels()):
           #  label.set_fontsize(5)
    cbaxes = fig.add_axes([1.0, 0.1, 0.01, 0.8]) 
    cbar = fig.colorbar(points, cax=cbaxes)
    cbar.set_label('Distance from Inlet(m)')
    cbar.set_clim(vmin=cmin, vmax=cmax)
    fig.tight_layout()

    #Methane v CO2
    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('GGA: Methane versus Carbon Dioxide', fontsize=18, y=1.08)
    plt.figure(1)
    axs = np.array(axs)
    for i in range(len(axs.reshape(-1))):
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['gga']['CH4_ppm_adjusted'], 
                                t[i]['gga']['CO2_ppm_adjusted'], 
                                c=t[i]['airmar']['distance'],
                                norm=norm, 
                                s=5, 
                                alpha=0.5, 
                                lw=0, 
                                cmap=cmap,
                                vmin=cmin,
                                vmax=cmax)
        axs[i].set_xlabel('Methane')
        axs[i].set_ylabel('CO2')

        # for label in (axs[i].get_xticklabels() + axs[i].get_yticklabels()):
           #  label.set_fontsize(5)
    cbaxes = fig.add_axes([1.0, 0.1, 0.01, 0.8]) 
    cbar = fig.colorbar(points, cax=cbaxes)
    cbar.set_label('Distance from Inlet(m)')
    cbar.set_clim(vmin=cmin, vmax=cmax)
    fig.tight_layout()

    #O2 versus Temeprature
    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('Optode: O2 Concentration versus Temperature', fontsize=18, y=1.08)
    plt.figure(1)
    axs = np.array(axs)
    for i in range(len(axs.reshape(-1))):
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['op']['O2Concentration'], 
                                t[i]['op']['Temperature'], 
                                c=t[i]['airmar']['distance'],
                                norm=norm, 
                                s=5, 
                                alpha=0.5, 
                                lw=0, 
                                cmap=cmap,
                                vmin=cmin,
                                vmax=cmax)
        axs[i].set_xlabel('O2')
        axs[i].set_ylabel('Temperature')

        # for label in (axs[i].get_xticklabels() + axs[i].get_yticklabels()):
           #  label.set_fontsize(5)
    cbaxes = fig.add_axes([1.0, 0.1, 0.01, 0.8]) 
    cbar = fig.colorbar(points, cax=cbaxes)
    cbar.set_label('Distance from Inlet(m)')
    cbar.set_clim(vmin=cmin, vmax=cmax)
    fig.tight_layout()

    #Salinity v Methane
    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('Salinity v Methane', fontsize=18, y=1.08)
    plt.figure(1)
    axs = np.array(axs)
    for i in range(len(axs.reshape(-1))):
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['ctd']['Salinity'], 
                                t[i]['gga']['CH4_ppm_adjusted'], 
                                c=t[i]['airmar']['distance'],
                                norm=norm, 
                                s=5, 
                                alpha=0.5, 
                                lw=0, 
                                cmap=cmap,
                                vmin=cmin,
                                vmax=cmax)
        axs[i].set_xlabel('Salinity')
        axs[i].set_ylabel('Methane')

        # for label in (axs[i].get_xticklabels() + axs[i].get_yticklabels()):
           #  label.set_fontsize(5)
    cbaxes = fig.add_axes([1.0, 0.1, 0.01, 0.8]) 
    cbar = fig.colorbar(points, cax=cbaxes)
    cbar.set_label('Distance from Inlet(m)')
    cbar.set_clim(vmin=cmin, vmax=cmax)
    fig.tight_layout()

    #Salinity v CO2
    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('Salinity v CO2', fontsize=18, y=1.08)
    plt.figure(1)
    axs = np.array(axs)
    for i in range(len(axs.reshape(-1))):
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['ctd']['Salinity'], 
                                t[i]['gga']['CO2_ppm_adjusted'], 
                                c=t[i]['airmar']['distance'],
                                norm=norm, 
                                s=5, 
                                alpha=0.5, 
                                lw=0, 
                                cmap=cmap,
                                vmin=cmin,
                                vmax=cmax)
        axs[i].set_xlabel('Salinity')
        axs[i].set_ylabel('CO2')

        # for label in (axs[i].get_xticklabels() + axs[i].get_yticklabels()):
           #  label.set_fontsize(5)
    cbaxes = fig.add_axes([1.0, 0.1, 0.01, 0.8]) 
    cbar = fig.colorbar(points, cax=cbaxes)
    cbar.set_label('Distance from Inlet(m)')
    cbar.set_clim(vmin=cmin, vmax=cmax)
    fig.tight_layout()

    #Salinity v O2
    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('Salinity v O2 Concentration', fontsize=18, y=1.08)
    plt.figure(1)
    axs = np.array(axs)
    for i in range(len(axs.reshape(-1))):
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['ctd']['Salinity'], 
                                t[i]['op']['O2Concentration'], 
                                c=t[i]['airmar']['distance'],
                                norm=norm, 
                                s=5, 
                                alpha=0.5, 
                                lw=0, 
                                cmap=cmap,
                                vmin=cmin,
                                vmax=cmax)
        axs[i].set_xlabel('Salinity')
        axs[i].set_ylabel('O2')

        # for label in (axs[i].get_xticklabels() + axs[i].get_yticklabels()):
           #  label.set_fontsize(5)
    cbaxes = fig.add_axes([1.0, 0.1, 0.01, 0.8]) 
    cbar = fig.colorbar(points, cax=cbaxes)
    cbar.set_label('Distance from Inlet(m)')
    cbar.set_clim(vmin=cmin, vmax=cmax)
    fig.tight_layout()

    #Salinity v Nitrate
    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('Salinity v Nitrate', fontsize=18, y=1.08)
    plt.figure(1)
    axs = np.array(axs)
    for i in range(len(axs.reshape(-1))):
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['ctd']['Salinity'], 
                                t[i]['nitrate']['0.00'], 
                                c=t[i]['airmar']['distance'],
                                norm=norm, 
                                s=5, 
                                alpha=0.5, 
                                lw=0, 
                                cmap=cmap,
                                vmin=cmin,
                                vmax=cmax)
        axs[i].set_xlabel('Salinity')
        axs[i].set_ylabel('Nitrate')

        # for label in (axs[i].get_xticklabels() + axs[i].get_yticklabels()):
           #  label.set_fontsize(5)
    cbaxes = fig.add_axes([1.0, 0.1, 0.01, 0.8]) 
    cbar = fig.colorbar(points, cax=cbaxes)
    cbar.set_label('Distance from Inlet(m)')
    cbar.set_clim(vmin=cmin, vmax=cmax)
    fig.tight_layout()

    #Temperature v Methane
    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('Temperature v Methane', fontsize=18, y=1.08)
    plt.figure(1)
    axs = np.array(axs)
    for i in range(len(axs.reshape(-1))):
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['ctd']['Temperature'], 
                                t[i]['gga']['CH4_ppm_adjusted'], 
                                c=t[i]['airmar']['distance'],
                                norm=norm, 
                                s=5, 
                                alpha=0.5, 
                                lw=0, 
                                cmap=cmap,
                                vmin=cmin,
                                vmax=cmax)
        axs[i].set_xlabel('Temperature')
        axs[i].set_ylabel('Methane')

        # for label in (axs[i].get_xticklabels() + axs[i].get_yticklabels()):
           #  label.set_fontsize(5)
    cbaxes = fig.add_axes([1.0, 0.1, 0.01, 0.8]) 
    cbar = fig.colorbar(points, cax=cbaxes)
    cbar.set_label('Distance from Inlet(m)')
    cbar.set_clim(vmin=cmin, vmax=cmax)
    fig.tight_layout()

    #Temperature v CO2
    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('Temperature v CO2', fontsize=18, y=1.08)
    plt.figure(1)
    axs = np.array(axs)
    for i in range(len(axs.reshape(-1))):
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['ctd']['Temperature'], 
                                t[i]['gga']['CO2_ppm_adjusted'], 
                                c=t[i]['airmar']['distance'],
                                norm=norm, 
                                s=5, 
                                alpha=0.5, 
                                lw=0, 
                                cmap=cmap,
                                vmin=cmin,
                                vmax=cmax)
        axs[i].set_xlabel('Temperature')
        axs[i].set_ylabel('CO2')

        # for label in (axs[i].get_xticklabels() + axs[i].get_yticklabels()):
           #  label.set_fontsize(5)
    cbaxes = fig.add_axes([1.0, 0.1, 0.01, 0.8]) 
    cbar = fig.colorbar(points, cax=cbaxes)
    cbar.set_label('Distance from Inlet(m)')
    cbar.set_clim(vmin=cmin, vmax=cmax)
    fig.tight_layout()

    #Temperature v O2
    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('Temperature v O2', fontsize=18, y=1.08)
    plt.figure(1)
    axs = np.array(axs)
    for i in range(len(axs.reshape(-1))):
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['ctd']['Temperature'], 
                                t[i]['op']['O2Concentration'], 
                                c=t[i]['airmar']['distance'],
                                norm=norm, 
                                s=5, 
                                alpha=0.5, 
                                lw=0, 
                                cmap=cmap,
                                vmin=cmin,
                                vmax=cmax)
        axs[i].set_xlabel('Temperature')
        axs[i].set_ylabel('O2')

        # for label in (axs[i].get_xticklabels() + axs[i].get_yticklabels()):
           #  label.set_fontsize(5)
    cbaxes = fig.add_axes([1.0, 0.1, 0.01, 0.8]) 
    cbar = fig.colorbar(points, cax=cbaxes)
    cbar.set_label('Distance from Inlet(m)')
    cbar.set_clim(vmin=cmin, vmax=cmax)
    fig.tight_layout()

    #Temperature v Nitrate
    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('Temperature v Nitrate', fontsize=18, y=1.08)
    plt.figure(1)
    axs = np.array(axs)
    for i in range(len(axs.reshape(-1))):
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['ctd']['Temperature'], 
                                t[i]['nitrate']['0.00'], 
                                c=t[i]['airmar']['distance'],
                                norm=norm, 
                                s=5, 
                                alpha=0.5, 
                                lw=0, 
                                cmap=cmap,
                                vmin=cmin,
                                vmax=cmax)
        axs[i].set_xlabel('Temperature')
        axs[i].set_ylabel('Nitrate')

        # for label in (axs[i].get_xticklabels() + axs[i].get_yticklabels()):
           #  label.set_fontsize(5)
    cbaxes = fig.add_axes([1.0, 0.1, 0.01, 0.8]) 
    cbar = fig.colorbar(points, cax=cbaxes)
    cbar.set_label('Distance from Inlet(m)')
    cbar.set_clim(vmin=cmin, vmax=cmax)
    fig.tight_layout()

    #Methane v O2
    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('Methane v O2', fontsize=18, y=1.08)
    plt.figure(1)
    axs = np.array(axs)
    for i in range(len(axs.reshape(-1))):
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['gga']['CH4_ppm_adjusted'], 
                                t[i]['op']['O2Concentration'], 
                                c=t[i]['airmar']['distance'],
                                norm=norm, 
                                s=5, 
                                alpha=0.5, 
                                lw=0, 
                                cmap=cmap,
                                vmin=cmin,
                                vmax=cmax)
        axs[i].set_xlabel('Methane')
        axs[i].set_ylabel('O2')

        # for label in (axs[i].get_xticklabels() + axs[i].get_yticklabels()):
           #  label.set_fontsize(5)
    cbaxes = fig.add_axes([1.0, 0.1, 0.01, 0.8]) 
    cbar = fig.colorbar(points, cax=cbaxes)
    cbar.set_label('Distance from Inlet(m)')
    cbar.set_clim(vmin=cmin, vmax=cmax)
    fig.tight_layout()

    #Methane v Nitrate
    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('Methane v Nitrate', fontsize=18, y=1.08)
    plt.figure(1)
    axs = np.array(axs)
    for i in range(len(axs.reshape(-1))):
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['gga']['CH4_ppm_adjusted'], 
                                t[i]['nitrate']['0.00'], 
                                c=t[i]['airmar']['distance'],
                                norm=norm, 
                                s=5, 
                                alpha=0.5, 
                                lw=0, 
                                cmap=cmap,
                                vmin=cmin,
                                vmax=cmax)
        axs[i].set_xlabel('Methane')
        axs[i].set_ylabel('Nitrate')

        # for label in (axs[i].get_xticklabels() + axs[i].get_yticklabels()):
           #  label.set_fontsize(5)
    cbaxes = fig.add_axes([1.0, 0.1, 0.01, 0.8]) 
    cbar = fig.colorbar(points, cax=cbaxes)
    cbar.set_label('Distance from Inlet(m)')
    cbar.set_clim(vmin=cmin, vmax=cmax)
    fig.tight_layout()

    #CO2 v O2
    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('CO2 v O2', fontsize=18, y=1.08)
    plt.figure(1)
    axs = np.array(axs)
    for i in range(len(axs.reshape(-1))):
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['gga']['CO2_ppm_adjusted'], 
                                t[i]['op']['O2Concentration'], 
                                c=t[i]['airmar']['distance'],
                                norm=norm, 
                                s=5, 
                                alpha=0.5, 
                                lw=0, 
                                cmap=cmap,
                                vmin=cmin,
                                vmax=cmax)
        axs[i].set_xlabel('CO2')
        axs[i].set_ylabel('O2')

        # for label in (axs[i].get_xticklabels() + axs[i].get_yticklabels()):
           #  label.set_fontsize(5)
    cbaxes = fig.add_axes([1.0, 0.1, 0.01, 0.8]) 
    cbar = fig.colorbar(points, cax=cbaxes)
    cbar.set_label('Distance from Inlet(m)')
    cbar.set_clim(vmin=cmin, vmax=cmax)
    fig.tight_layout()

    #CO2 v Nitrate
    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('CO2 v Nitrate', fontsize=18, y=1.08)
    plt.figure(1)
    axs = np.array(axs)
    for i in range(len(axs.reshape(-1))):
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['gga']['CO2_ppm_adjusted'], 
                                t[i]['nitrate']['0.00'], 
                                c=t[i]['airmar']['distance'],
                                norm=norm, 
                                s=5, 
                                alpha=0.5, 
                                lw=0, 
                                cmap=cmap,
                                vmin=cmin,
                                vmax=cmax)
        axs[i].set_xlabel('CO2')
        axs[i].set_ylabel('Nitrate')

        # for label in (axs[i].get_xticklabels() + axs[i].get_yticklabels()):
           #  label.set_fontsize(5)
    cbaxes = fig.add_axes([1.0, 0.1, 0.01, 0.8]) 
    cbar = fig.colorbar(points, cax=cbaxes)
    cbar.set_label('Distance from Inlet(m)')
    cbar.set_clim(vmin=cmin, vmax=cmax)
    fig.tight_layout()

    #Nitrate v O2
    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('Nitrate v O2', fontsize=18, y=1.08)
    plt.figure(1)
    axs = np.array(axs)
    for i in range(len(axs.reshape(-1))):
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['nitrate']['0.00'], 
                                t[i]['op']['O2Concentration'], 
                                c=t[i]['airmar']['distance'],
                                norm=norm, 
                                s=5, 
                                alpha=0.5, 
                                lw=0, 
                                cmap=cmap,
                                vmin=cmin,
                                vmax=cmax)
        axs[i].set_xlabel('Nitrate')
        axs[i].set_ylabel('O2')

        # for label in (axs[i].get_xticklabels() + axs[i].get_yticklabels()):
           #  label.set_fontsize(5)
    cbaxes = fig.add_axes([1.0, 0.1, 0.01, 0.8]) 
    cbar = fig.colorbar(points, cax=cbaxes)
    cbar.set_label('Distance from Inlet(m)')
    cbar.set_clim(vmin=cmin, vmax=cmax)
    fig.tight_layout()

def plot_transect_pvprange_master(ta, tb, tc, td, te):
    t = [ta, tb, tc, td, te]
    labels = ['A', 'B', 'C', 'D', 'E']
    c=[]
    for transect in t:
        c.append(['red' if x == 1.0 else 'blue' for x in transect['airmar']['in_range']])
    
    #Salinity vs Temperature
    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('CTD: Salinity versus Temperature', fontsize=18, y=1.08)
    plt.figure(1)
    axs = np.array(axs)
    for i in range(len(axs.reshape(-1))):
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['ctd']['Salinity'], 
                                t[i]['ctd']['Temperature'], 
                                c=c[i],
                                s=5, 
                                alpha=0.5, 
                                lw=0)
        axs[i].set_xlabel('Salinity')
        axs[i].set_ylabel('Temperature')
    patch = mpatches.Patch(color='red', label='200m range of plume')
    patch2 = mpatches.Patch(color='blue', label='Outside 200m')
    fig.legend( [patch, patch2], ['200m range of plume', 'Outside 200m'], loc = 'lower right', ncol=2, labelspacing=3.0, borderaxespad=3.0, fontsize=18)
    fig.tight_layout()

    #Methane v CO2
    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('GGA: Methane versus Carbon Dioxide', fontsize=18, y=1.08)
    plt.figure(1)
    axs = np.array(axs)
    for i in range(len(axs.reshape(-1))):
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['gga']['CH4_ppm_adjusted'], 
                                t[i]['gga']['CO2_ppm_adjusted'], 
                                c=c[i],
                                s=5, 
                                alpha=0.5, 
                                lw=0)
        axs[i].set_xlabel('Methane')
        axs[i].set_ylabel('CO2')
    patch = mpatches.Patch(color='red', label='200m range of plume')
    patch2 = mpatches.Patch(color='blue', label='Outside 200m')
    fig.legend( [patch, patch2], ['200m range of plume', 'Outside 200m'], loc = 'lower right', ncol=2, labelspacing=3.0, borderaxespad=3.0, fontsize=18)
    fig.tight_layout()

    # #O2 versus Temeprature
    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('Optode: O2 Concentration versus Temperature', fontsize=18, y=1.08)
    plt.figure(1)
    axs = np.array(axs)
    for i in range(len(axs.reshape(-1))):
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['op']['O2Concentration'], 
                                t[i]['op']['Temperature'], 
                                c=c[i],
                                s=5, 
                                alpha=0.5, 
                                lw=0)
        axs[i].set_xlabel('O2')
        axs[i].set_ylabel('Temperature')
    patch = mpatches.Patch(color='red', label='200m range of plume')
    patch2 = mpatches.Patch(color='blue', label='Outside 200m')
    fig.legend( [patch, patch2], ['200m range of plume', 'Outside 200m'], loc = 'lower right', ncol=2, labelspacing=3.0, borderaxespad=3.0, fontsize=18)
    fig.tight_layout()

    #Salinity v Methane
    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('Salinity v Methane', fontsize=18, y=1.08)
    plt.figure(1)
    axs = np.array(axs)
    for i in range(len(axs.reshape(-1))):
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['ctd']['Salinity'], 
                                t[i]['gga']['CH4_ppm_adjusted'], 
                                c=c[i],
                                s=5, 
                                alpha=0.5, 
                                lw=0)
        axs[i].set_xlabel('Salinity')
        axs[i].set_ylabel('Methane')
    patch = mpatches.Patch(color='red', label='200m range of plume')
    patch2 = mpatches.Patch(color='blue', label='Outside 200m')
    fig.legend( [patch, patch2], ['200m range of plume', 'Outside 200m'], loc = 'lower right', ncol=2, labelspacing=3.0, borderaxespad=3.0, fontsize=18)
    fig.tight_layout()

    # #Salinity v CO2
    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('Salinity v CO2', fontsize=18, y=1.08)
    plt.figure(1)
    axs = np.array(axs)
    for i in range(len(axs.reshape(-1))):
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['ctd']['Salinity'], 
                                t[i]['gga']['CO2_ppm_adjusted'], 
                                c=c[i],
                                s=5, 
                                alpha=0.5, 
                                lw=0)
        axs[i].set_xlabel('Salinity')
        axs[i].set_ylabel('CO2')
    patch = mpatches.Patch(color='red', label='200m range of plume')
    patch2 = mpatches.Patch(color='blue', label='Outside 200m')
    fig.legend( [patch, patch2], ['200m range of plume', 'Outside 200m'], loc = 'lower right', ncol=2, labelspacing=3.0, borderaxespad=3.0, fontsize=18)
    fig.tight_layout()

    #Salinity v O2
    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('Salinity v O2 Concentration', fontsize=18, y=1.08)
    plt.figure(1)
    axs = np.array(axs)
    for i in range(len(axs.reshape(-1))):
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['ctd']['Salinity'], 
                                t[i]['op']['O2Concentration'], 
                                c=c[i],
                                s=5, 
                                alpha=0.5, 
                                lw=0)
        axs[i].set_xlabel('Salinity')
        axs[i].set_ylabel('O2')
    patch = mpatches.Patch(color='red', label='200m range of plume')
    patch2 = mpatches.Patch(color='blue', label='Outside 200m')
    fig.legend( [patch, patch2], ['200m range of plume', 'Outside 200m'], loc = 'lower right', ncol=2, labelspacing=3.0, borderaxespad=3.0, fontsize=18)
    fig.tight_layout()

    #Salinity v Nitrate
    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('Salinity v Nitrate', fontsize=18, y=1.08)
    plt.figure(1)
    axs = np.array(axs)
    for i in range(len(axs.reshape(-1))):
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['ctd']['Salinity'], 
                                t[i]['nitrate']['0.00'], 
                                c=c[i],
                                s=5, 
                                alpha=0.5, 
                                lw=0)
        axs[i].set_xlabel('Salinity')
        axs[i].set_ylabel('Nitrate')
    patch = mpatches.Patch(color='red', label='200m range of plume')
    patch2 = mpatches.Patch(color='blue', label='Outside 200m')
    fig.legend( [patch, patch2], ['200m range of plume', 'Outside 200m'], loc = 'lower right', ncol=2, labelspacing=3.0, borderaxespad=3.0, fontsize=18)
    fig.tight_layout()

    #Temperature v Methane
    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('Temperature v Methane', fontsize=18, y=1.08)
    plt.figure(1)
    axs = np.array(axs)
    for i in range(len(axs.reshape(-1))):
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['ctd']['Temperature'], 
                                t[i]['gga']['CH4_ppm_adjusted'], 
                                c=c[i],
                                s=5, 
                                alpha=0.5, 
                                lw=0)
        axs[i].set_xlabel('Temperature')
        axs[i].set_ylabel('Methane')
    patch = mpatches.Patch(color='red', label='200m range of plume')
    patch2 = mpatches.Patch(color='blue', label='Outside 200m')
    fig.legend( [patch, patch2], ['200m range of plume', 'Outside 200m'], loc = 'lower right', ncol=2, labelspacing=3.0, borderaxespad=3.0, fontsize=18)
    fig.tight_layout()

    #Temperature v CO2
    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('Temperature v CO2', fontsize=18, y=1.08)
    plt.figure(1)
    axs = np.array(axs)
    for i in range(len(axs.reshape(-1))):
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['ctd']['Temperature'], 
                                t[i]['gga']['CO2_ppm_adjusted'], 
                                c=c[i],
                                s=5, 
                                alpha=0.5, 
                                lw=0)
        axs[i].set_xlabel('Temperature')
        axs[i].set_ylabel('CO2')
    patch = mpatches.Patch(color='red', label='200m range of plume')
    patch2 = mpatches.Patch(color='blue', label='Outside 200m')
    fig.legend( [patch, patch2], ['200m range of plume', 'Outside 200m'], loc = 'lower right', ncol=2, labelspacing=3.0, borderaxespad=3.0, fontsize=18)
    fig.tight_layout()

    #Temperature v O2
    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('Temperature v O2', fontsize=18, y=1.08)
    plt.figure(1)
    axs = np.array(axs)
    for i in range(len(axs.reshape(-1))):
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['ctd']['Temperature'], 
                                t[i]['op']['O2Concentration'], 
                                c=c[i],
                                s=5, 
                                alpha=0.5, 
                                lw=0)
        axs[i].set_xlabel('Temperature')
        axs[i].set_ylabel('O2')
    patch = mpatches.Patch(color='red', label='200m range of plume')
    patch2 = mpatches.Patch(color='blue', label='Outside 200m')
    fig.legend( [patch, patch2], ['200m range of plume', 'Outside 200m'], loc = 'lower right', ncol=2, labelspacing=3.0, borderaxespad=3.0, fontsize=18)
    fig.tight_layout()

    #Temperature v Nitrate
    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('Temperature v Nitrate', fontsize=18, y=1.08)
    plt.figure(1)
    axs = np.array(axs)
    for i in range(len(axs.reshape(-1))):
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['ctd']['Temperature'], 
                                t[i]['nitrate']['0.00'], 
                                c=c[i],
                                s=5, 
                                alpha=0.5, 
                                lw=0)
        axs[i].set_xlabel('Temperature')
        axs[i].set_ylabel('Nitrate')
    patch = mpatches.Patch(color='red', label='200m range of plume')
    patch2 = mpatches.Patch(color='blue', label='Outside 200m')
    fig.legend( [patch, patch2], ['200m range of plume', 'Outside 200m'], loc = 'lower right', ncol=2, labelspacing=3.0, borderaxespad=3.0, fontsize=18)
    fig.tight_layout()

    #Methane v O2
    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('Methane v O2', fontsize=18, y=1.08)
    plt.figure(1)
    axs = np.array(axs)
    for i in range(len(axs.reshape(-1))):
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['gga']['CH4_ppm_adjusted'], 
                                t[i]['op']['O2Concentration'], 
                                c=c[i],
                                s=5, 
                                alpha=0.5, 
                                lw=0)
        axs[i].set_xlabel('Methane')
        axs[i].set_ylabel('O2')
    patch = mpatches.Patch(color='red', label='200m range of plume')
    patch2 = mpatches.Patch(color='blue', label='Outside 200m')
    fig.legend( [patch, patch2], ['200m range of plume', 'Outside 200m'], loc = 'lower right', ncol=2, labelspacing=3.0, borderaxespad=3.0, fontsize=18)
    fig.tight_layout()

    #Methane v Nitrate
    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('Methane v Nitrate', fontsize=18, y=1.08)
    plt.figure(1)
    axs = np.array(axs)
    for i in range(len(axs.reshape(-1))):
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['gga']['CH4_ppm_adjusted'], 
                                t[i]['nitrate']['0.00'], 
                                c=c[i],
                                s=5, 
                                alpha=0.5, 
                                lw=0)
        axs[i].set_xlabel('Methane')
        axs[i].set_ylabel('Nitrate')
    patch = mpatches.Patch(color='red', label='200m range of plume')
    patch2 = mpatches.Patch(color='blue', label='Outside 200m')
    fig.legend( [patch, patch2], ['200m range of plume', 'Outside 200m'], loc = 'lower right', ncol=2, labelspacing=3.0, borderaxespad=3.0, fontsize=18)
    fig.tight_layout()

    #CO2 v O2
    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('CO2 v O2', fontsize=18, y=1.08)
    plt.figure(1)
    axs = np.array(axs)
    for i in range(len(axs.reshape(-1))):
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['gga']['CO2_ppm_adjusted'], 
                                t[i]['op']['O2Concentration'], 
                                c=c[i],
                                s=5, 
                                alpha=0.5, 
                                lw=0)
        axs[i].set_xlabel('CO2')
        axs[i].set_ylabel('O2')
    patch = mpatches.Patch(color='red', label='200m range of plume')
    patch2 = mpatches.Patch(color='blue', label='Outside 200m')
    fig.legend( [patch, patch2], ['200m range of plume', 'Outside 200m'], loc = 'lower right', ncol=2, labelspacing=3.0, borderaxespad=3.0, fontsize=18)
    fig.tight_layout()

    #CO2 v Nitrate
    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('CO2 v Nitrate', fontsize=18, y=1.08)
    plt.figure(1)
    axs = np.array(axs)
    for i in range(len(axs.reshape(-1))):
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['gga']['CO2_ppm_adjusted'], 
                                t[i]['nitrate']['0.00'], 
                                c=c[i],
                                s=5, 
                                alpha=0.5, 
                                lw=0)
        axs[i].set_xlabel('CO2')
        axs[i].set_ylabel('Nitrate')
    patch = mpatches.Patch(color='red', label='200m range of plume')
    patch2 = mpatches.Patch(color='blue', label='Outside 200m')
    fig.legend( [patch, patch2], ['200m range of plume', 'Outside 200m'], loc = 'lower right', ncol=2, labelspacing=3.0, borderaxespad=3.0, fontsize=18)
    fig.tight_layout()

    #Nitrate v O2
    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('Nitrate v O2', fontsize=18, y=1.08)
    plt.figure(1)
    axs = np.array(axs)
    for i in range(len(axs.reshape(-1))):
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['nitrate']['0.00'], 
                                t[i]['op']['O2Concentration'], 
                                c=c[i],
                                s=5, 
                                alpha=0.5, 
                                lw=0)
        axs[i].set_xlabel('Nitrate')
        axs[i].set_ylabel('O2')
    patch = mpatches.Patch(color='red', label='200m range of plume')
    patch2 = mpatches.Patch(color='blue', label='Outside 200m')
    fig.legend( [patch, patch2], ['200m range of plume', 'Outside 200m'], loc = 'lower right', ncol=2, labelspacing=3.0, borderaxespad=3.0, fontsize=18)
    fig.tight_layout()

######
#Spatial Graphs
######
def plot_bilinearinterp(x, y, data, bins, label, title):
    weighted, _, _ = np.histogram2d(x, y, weights=data, normed=False, bins=bins)
    count, xedges, yedges = np.histogram2d(x,y,bins=bins)
    Z = weighted/count
    
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2

    fig, axs = plt.subplots()
    fig.subplots_adjust(bottom=0.07, hspace=0.3)
    fig.suptitle('Bilinear Interpolation of ' + title, fontsize=18)

    polys  = shapefile.Reader('./GIS_Data/River.shp')
    poly = polys.iterShapes().next().__geo_interface__
    patch = PolygonPatch(poly, fc='green', ec="#6688cc", alpha=0.3, zorder=0)
    axs.add_patch(patch)
    axs.axis(xmin=-70.691185, ymin=41.756619, xmax=-70.675049, ymax=41.765134)

    im = NonUniformImage(axs, interpolation='nearest', extent=(-70.691185, -70.675049, 41.756619, 41.765134),
                         cmap=cm.bwr)
    im.set_data(xcenters, ycenters, Z.T)
    im.set_clip_path(patch)
    axs.images.append(im)
    axs.set_xlim(-70.691185, -70.6780)
    axs.set_ylim(41.756619, 41.765134)

    cbar = fig.colorbar(im)
    cbar.set_label(label)

    plt.show()

def plot_contours(x, y, data, bins, label, title):
    weighted, _, _ = np.histogram2d(x, y, weights=data, normed=False, bins=bins)
    count, xedges, yedges = np.histogram2d(x,y,bins=bins)
    Z = weighted/count
    
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(xcenters, ycenters)

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.07, hspace=0.3)
    fig.suptitle('Contours of ' + title, fontsize=18)

    cs = ax.contourf(X, Y, Z.T, cmap=plt.cm.bwr)

    polys  = shapefile.Reader('./GIS_Data/River.shp')
    poly = polys.iterShapes().next().__geo_interface__
    patch = PolygonPatch(poly, fc='green', ec="#6688cc", alpha=0.3, zorder=0)
    ax.add_patch(patch)
    ax.axis(xmin=-70.691185, ymin=41.756619, xmax=-70.678, ymax=41.765134)

    cbar = fig.colorbar(cs)
    cbar.set_label(label)

    for collection in cs.collections:
        collection.set_clip_path(patch)

def plot_rawspatial(x, y, data, label, title):
    map_c = plt.figure()
    ax = map_c.gca()

    polys  = shapefile.Reader('./GIS_Data/River.shp')
    poly = polys.iterShapes().next().__geo_interface__
    patch = PolygonPatch(poly, fc='green', ec="#6688cc", alpha=0.3, zorder=0)
    ax.add_patch(patch)
    ax.axis(xmin=-70.691185, ymin=41.756619, xmax=-70.678, ymax=41.765134)

    cmap = plt.cm.bwr
    points = ax.scatter(x, y, c=data, s=5, alpha=1.0, lw=0, cmap=cmap)
    cbar = map_c.colorbar(points)
    cbar.set_label(label)
    points.set_clip_path(patch)

    map_c.suptitle('Raw Plotted Data for ' + title, fontsize=18)
    plt.show()

def plot_interptransects(df, transect_name, bins):
    #Salinity
    x = df['airmar']['lon_mod'][1:]
    y = df['airmar']['lat_mod'][1:]
    bins = bins
    
    salinity = df['ctd']['Salinity'][1:]
    label = 'Salinity Level'
    title = 'Salinity Across ' + transect_name
    plot_rawspatial(x, y, salinity, label, title)
    # plot_bilinearinterp(x, y, salinity, bins, label, title)
    # plot_contours(x, y, salinity, bins, label, title)

    temperature = df['ctd']['Temperature'][1:]
    label = 'Temperaure Level'
    title = 'Temperaure Across ' + transect_name
    plot_rawspatial(x, y, temperature, label, title)
    # plot_bilinearinterp(x, y, temperature, bins, label, title)
    # plot_contours(x, y, temperature, bins, label, title)

    methane = df['gga']['CH4_ppm_adjusted'][1:]
    label = 'Methane Level'
    title = 'Methane Across ' + transect_name
    plot_rawspatial(x, y, methane, label, title)
    # plot_bilinearinterp(x, y, methane, bins, label, title)
    # plot_contours(x, y, methane, bins, label, title)

    co2 = df['gga']['CO2_ppm_adjusted'][1:]
    label = 'CO2 Level'
    title = 'CO2 Across ' + transect_name
    plot_rawspatial(x, y, co2, label, title)
    # plot_bilinearinterp(x, y, co2, bins, label, title)
    # plot_contours(x, y, co2, bins, label, title)

    oxy = df['op']['O2Concentration'][1:]
    label = 'Oxygen Level'
    title = 'Oxygen Across ' + transect_name
    plot_rawspatial(x, y, oxy, label, title)
    # plot_bilinearinterp(x, y, oxy, bins, label, title)
    # plot_contours(x, y, oxy, bins, label, title)

    nit = df['nitrate']['0.00'][1:]
    label = 'Nitrate Level'
    title = 'Nitrate Across ' + transect_name
    plot_rawspatial(x, y, nit, label, title)
    # plot_bilinearinterp(x, y, nit, bins, label, title)
    # plot_contours(x, y, nit, bins, label, title)

def plot_master(df, ta, tb, tc, td, te):
    polys  = shapefile.Reader('./GIS_Data/River.shp')
    poly = polys.iterShapes().next().__geo_interface__

    t = [ta, tb, tc, td, te]
    labels = ['A', 'B', 'C', 'D', 'E']

    #Salinity
    cmin = df['ctd']['Salinity'].min()
    cmax = df['ctd']['Salinity'].max()
    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('Salinity', fontsize=18)
    axs = np.array(axs)

    cmap = plt.cm.bwr

    for i in range(len(axs.reshape(-1))):
        axs[i].add_patch(PolygonPatch(poly, fc='green', ec="#6688cc", alpha=0.3, zorder=0))
        axs[i].axis(xmin=-70.691185, ymin=41.756619, xmax=-70.678, ymax=41.765134)
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['airmar']['lon_mod'], 
                                t[i]['airmar']['lat_mod'], 
                                c=t[i]['ctd']['Salinity'], 
                                s=5, 
                                alpha=1.0, 
                                lw=0, 
                                cmap=cmap, 
                                vmin=cmin, 
                                vmax=cmax)

        for label in (axs[i].get_xticklabels() + axs[i].get_yticklabels()):
            label.set_fontsize(5)

    cbaxes = fig.add_axes([1.0, 0.1, 0.01, 0.8]) 
    cbar = fig.colorbar(points, cax=cbaxes)
    cbar.set_label('Salinity Level')
    cbar.set_clim(vmin=cmin, vmax=cmax)
    fig.tight_layout()

    #Temperature
    cmin = df['ctd']['Temperature'].min()
    cmax = df['ctd']['Temperature'].max()

    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('Temperature', fontsize=18)
    axs = np.array(axs)

    cmap = plt.cm.bwr

    for i in range(len(axs.reshape(-1))):
        axs[i].add_patch(PolygonPatch(poly, fc='green', ec="#6688cc", alpha=0.3, zorder=0))
        axs[i].axis(xmin=-70.691185, ymin=41.756619, xmax=-70.678, ymax=41.765134)
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['airmar']['lon_mod'], 
                                t[i]['airmar']['lat_mod'], 
                                c=t[i]['ctd']['Temperature'], 
                                s=5, 
                                alpha=1.0, 
                                lw=0, 
                                cmap=cmap, 
                                vmin=cmin, 
                                vmax=cmax)
        for label in (axs[i].get_xticklabels() + axs[i].get_yticklabels()):
            label.set_fontsize(5)

    cbaxes = fig.add_axes([1.0, 0.1, 0.01, 0.8]) 
    cbar = fig.colorbar(points, cax=cbaxes)
    cbar.set_label('Temperature Level')
    cbar.set_clim(vmin=cmin, vmax=cmax)
    fig.tight_layout()

    #Methane
    cmin = df['gga']['CH4_ppm_adjusted'].min()
    cmax = 200#df['gga']['CH4_ppm_adjusted'].max()

    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('Methane', fontsize=18)
    axs = np.array(axs)

    cmap = plt.cm.bwr

    for i in range(len(axs.reshape(-1))):
        axs[i].add_patch(PolygonPatch(poly, fc='green', ec="#6688cc", alpha=0.3, zorder=0))
        axs[i].axis(xmin=-70.691185, ymin=41.756619, xmax=-70.678, ymax=41.765134)
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['airmar']['lon_mod'], 
                                t[i]['airmar']['lat_mod'], 
                                c=t[i]['gga']['CH4_ppm_adjusted'], 
                                s=5, 
                                alpha=1.0, 
                                lw=0, 
                                cmap=cmap, 
                                vmin=cmin, 
                                vmax=cmax)
        for label in (axs[i].get_xticklabels() + axs[i].get_yticklabels()):
            label.set_fontsize(5)

    cbaxes = fig.add_axes([1.0, 0.1, 0.01, 0.8]) 
    cbar = fig.colorbar(points, cax=cbaxes)
    cbar.set_label('Methane Level')
    cbar.set_clim(vmin=cmin, vmax=cmax)
    fig.tight_layout()

    #CO2
    cmin = df['gga']['CO2_ppm_adjusted'].min()
    cmax = df['gga']['CO2_ppm_adjusted'].max()

    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('Carbon Dioxide', fontsize=18)
    axs = np.array(axs)

    cmap = plt.cm.bwr

    for i in range(len(axs.reshape(-1))):
        axs[i].add_patch(PolygonPatch(poly, fc='green', ec="#6688cc", alpha=0.3, zorder=0))
        axs[i].axis(xmin=-70.691185, ymin=41.756619, xmax=-70.678, ymax=41.765134)
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['airmar']['lon_mod'], 
                                t[i]['airmar']['lat_mod'], 
                                c=t[i]['gga']['CO2_ppm_adjusted'], 
                                s=5, 
                                alpha=1.0, 
                                lw=0, 
                                cmap=cmap, 
                                vmin=cmin, 
                                vmax=cmax)
        for label in (axs[i].get_xticklabels() + axs[i].get_yticklabels()):
            label.set_fontsize(5)

    cbaxes = fig.add_axes([1.0, 0.1, 0.01, 0.8]) 
    cbar = fig.colorbar(points, cax=cbaxes)
    cbar.set_label('CO2 Level')
    cbar.set_clim(vmin=cmin, vmax=cmax)
    fig.tight_layout()

    #O2
    cmin = df['op']['O2Concentration'].min()
    cmax = df['op']['O2Concentration'].max()

    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('Oxygen', fontsize=18)
    axs = np.array(axs)

    cmap = plt.cm.bwr

    for i in range(len(axs.reshape(-1))):
        axs[i].add_patch(PolygonPatch(poly, fc='green', ec="#6688cc", alpha=0.3, zorder=0))
        axs[i].axis(xmin=-70.691185, ymin=41.756619, xmax=-70.678, ymax=41.765134)
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['airmar']['lon_mod'], 
                                t[i]['airmar']['lat_mod'], 
                                c=t[i]['op']['O2Concentration'], 
                                s=5, 
                                alpha=1.0, 
                                lw=0, 
                                cmap=cmap, 
                                vmin=cmin, 
                                vmax=cmax)
        for label in (axs[i].get_xticklabels() + axs[i].get_yticklabels()):
            label.set_fontsize(5)

    cbaxes = fig.add_axes([1.0, 0.1, 0.01, 0.8]) 
    cbar = fig.colorbar(points, cax=cbaxes)
    cbar.set_label('Oxygen Level')
    cbar.set_clim(vmin=cmin, vmax=cmax)
    fig.tight_layout()

    #Nitrate
    cmin = df['nitrate']['0.00'].min()
    cmax = df['nitrate']['0.00'].max()

    fig, axs = plt.subplots(1,5, figsize=(20,5), sharex='row', sharey='row')
    fig.suptitle('Nitrate', fontsize=18)
    axs = np.array(axs)

    cmap = plt.cm.bwr

    for i in range(len(axs.reshape(-1))):
        axs[i].add_patch(PolygonPatch(poly, fc='green', ec="#6688cc", alpha=0.3, zorder=0))
        axs[i].axis(xmin=-70.691185, ymin=41.756619, xmax=-70.678, ymax=41.765134)
        axs[i].set_title(labels[i])
        points = axs[i].scatter(t[i]['airmar']['lon_mod'], 
                                t[i]['airmar']['lat_mod'], 
                                c=t[i]['nitrate']['0.00'], 
                                s=5, 
                                alpha=1.0, 
                                lw=0, 
                                cmap=cmap, 
                                vmin=cmin, 
                                vmax=cmax)
        for label in (axs[i].get_xticklabels() + axs[i].get_yticklabels()):
            label.set_fontsize(5)

    cbaxes = fig.add_axes([1.0, 0.1, 0.01, 0.8]) 
    cbar = fig.colorbar(points, cax=cbaxes)
    cbar.set_label('Nitrate Level')
    cbar.set_clim(vmin=cmin, vmax=cmax)
    fig.tight_layout()

    plt.plot()

######
#Distance Graphs
######
def plot_averages(ta, tb, tc, td, te, window):
    #create new dataframes
    newa = pd.concat([ta['airmar']['distance'][1:],
                      ta['ctd']['Salinity'][1:],
                      ta['ctd']['Temperature'][1:],
                      ta['gga']['CH4_ppm_adjusted'][1:],
                      ta['gga']['CO2_ppm_adjusted'][1:],
                      ta['op']['O2Concentration'][1:],
                      ta['nitrate']['0.00'][1:]],
                      axis=1)

    newb = pd.concat([tb['airmar']['distance'][1:],
                      tb['ctd']['Salinity'][1:],
                      tb['ctd']['Temperature'][1:],
                      tb['gga']['CH4_ppm_adjusted'][1:],
                      tb['gga']['CO2_ppm_adjusted'][1:],
                      tb['op']['O2Concentration'][1:],
                      tb['nitrate']['0.00'][1:]],
                      axis=1)

    newc = pd.concat([tc['airmar']['distance'][1:],
                      tc['ctd']['Salinity'][1:],
                      tc['ctd']['Temperature'][1:],
                      tc['gga']['CH4_ppm_adjusted'][1:],
                      tc['gga']['CO2_ppm_adjusted'][1:],
                      tc['op']['O2Concentration'][1:],
                      tc['nitrate']['0.00'][1:]],
                      axis=1)

    newd = pd.concat([td['airmar']['distance'][1:],
                      td['ctd']['Salinity'][1:],
                      td['ctd']['Temperature'][1:],
                      td['gga']['CH4_ppm_adjusted'][1:],
                      td['gga']['CO2_ppm_adjusted'][1:],
                      td['op']['O2Concentration'][1:],
                      td['nitrate']['0.00'][1:]],
                      axis=1)

    newe = pd.concat([te['airmar']['distance'][1:],
                      te['ctd']['Salinity'][1:],
                      te['ctd']['Temperature'][1:],
                      te['gga']['CH4_ppm_adjusted'][1:],
                      te['gga']['CO2_ppm_adjusted'][1:],
                      te['op']['O2Concentration'][1:],
                      te['nitrate']['0.00'][1:]],
                      axis=1)


    #Sort based on distance
    newa = newa.sort_values('distance')
    newb = newb.sort_values('distance')
    newc = newc.sort_values('distance')
    newd = newd.sort_values('distance')
    newe = newe.sort_values('distance')

    #Average
    ra = newa.rolling(window=window)
    rb = newb.rolling(window=window)
    rc = newc.rolling(window=window)
    rd = newd.rolling(window=window)
    re = newe.rolling(window=window)

    #Plot
    plt.figure(1)
    plt.title('Average Salinity')
    plt.xlabel('Distance from Effluent Inlet (m)')
    plt.ylabel('Salinity Level')
    plt.plot(ra['distance'].mean(), ra['Salinity'].mean(), 'b-', label='TransectA')
    plt.plot(rb['distance'].mean(), rb['Salinity'].mean(), 'r-', label='TransectB')
    plt.plot(rc['distance'].mean(), rc['Salinity'].mean(), 'g-', label='TransectC')
    plt.plot(rd['distance'].mean(), rd['Salinity'].mean(), 'k-', label='TransectD')
    plt.plot(re['distance'].mean(), re['Salinity'].mean(), 'y-', label='TransectE')
    plt.legend()

    plt.figure(2)
    plt.title('Average Temperature')
    plt.xlabel('Distance from Effluent Inlet (m)')
    plt.ylabel('Temperature (C)')
    plt.plot(ra['distance'].mean(), ra['Temperature'].mean(), 'b-', label='TransectA')
    plt.plot(rb['distance'].mean(), rb['Temperature'].mean(), 'r-', label='TransectB')
    plt.plot(rc['distance'].mean(), rc['Temperature'].mean(), 'g-', label='TransectC')
    plt.plot(rd['distance'].mean(), rd['Temperature'].mean(), 'k-', label='TransectD')
    plt.plot(re['distance'].mean(), re['Temperature'].mean(), 'y-', label='TransectE')
    plt.legend()

    plt.figure(3)
    plt.title('Average Methane')
    plt.xlabel('Distance from Effluent Inlet (m)')
    plt.ylabel('Methane Level (uatm)')
    plt.plot(ra['distance'].mean(), ra['CH4_ppm_adjusted'].mean(), 'b-', label='TransectA')
    plt.plot(rb['distance'].mean(), rb['CH4_ppm_adjusted'].mean(), 'r-', label='TransectB')
    plt.plot(rc['distance'].mean(), rc['CH4_ppm_adjusted'].mean(), 'g-', label='TransectC')
    plt.plot(rd['distance'].mean(), rd['CH4_ppm_adjusted'].mean(), 'k-', label='TransectD')
    plt.plot(re['distance'].mean(), re['CH4_ppm_adjusted'].mean(), 'y-', label='TransectE')
    plt.legend()

    plt.figure(4)
    plt.title('Average CO2')
    plt.xlabel('Distance from Effluent Inlet (m)')
    plt.ylabel('CO2 Level (uatm)')
    plt.plot(ra['distance'].mean(), ra['CO2_ppm_adjusted'].mean(), 'b-', label='TransectA')
    plt.plot(rb['distance'].mean(), rb['CO2_ppm_adjusted'].mean(), 'r-', label='TransectB')
    plt.plot(rc['distance'].mean(), rc['CO2_ppm_adjusted'].mean(), 'g-', label='TransectC')
    plt.plot(rd['distance'].mean(), rd['CO2_ppm_adjusted'].mean(), 'k-', label='TransectD')
    plt.plot(re['distance'].mean(), re['CO2_ppm_adjusted'].mean(), 'y-', label='TransectE')
    plt.legend()

    plt.figure(5)
    plt.title('Average 02')
    plt.xlabel('Distance from Effluent Inlet (m)')
    plt.ylabel('02 Level')
    plt.plot(ra['distance'].mean(), ra['O2Concentration'].mean(), 'b-', label='TransectA')
    plt.plot(rb['distance'].mean(), rb['O2Concentration'].mean(), 'r-', label='TransectB')
    plt.plot(rc['distance'].mean(), rc['O2Concentration'].mean(), 'g-', label='TransectC')
    plt.plot(rd['distance'].mean(), rd['O2Concentration'].mean(), 'k-', label='TransectD')
    plt.plot(re['distance'].mean(), re['O2Concentration'].mean(), 'y-', label='TransectE')
    plt.legend()

    plt.figure(6)
    plt.title('Average Nitrate')
    plt.xlabel('Distance from Effluent Inlet (m)')
    plt.ylabel('Nitrate Level')
    plt.plot(ra['distance'].mean(), ra['0.00'].mean(), 'b-', label='TransectA')
    plt.plot(rb['distance'].mean(), rb['0.00'].mean(), 'r-', label='TransectB')
    plt.plot(rc['distance'].mean(), rc['0.00'].mean(), 'g-', label='TransectC')
    plt.plot(rd['distance'].mean(), rd['0.00'].mean(), 'k-', label='TransectD')
    plt.plot(re['distance'].mean(), re['0.00'].mean(), 'y-', label='TransectE')
    plt.legend()

    plt.show()

def plot_distance_master(ta, tb, tc, td, te):
    t = [ta, tb, tc, td, te]
    labels = ['A', 'B', 'C', 'D', 'E']
    instruments = ['ctd', 'ctd', 'gga', 'gga', 'nitrate', 'op']
    measures = ['Salinity', 'Temperature', 'CH4_ppm_adjusted', 'CO2_ppm_adjusted', '0.00', 'O2Concentration']
    mlabels=['Salinity', 'Temperature', 'CH4', 'CO2', 'Nitrate', 'O2']

    fig, axs = plt.subplots(6,5, figsize=(40,40), sharex='col', sharey='row')
    fig.suptitle('Distance Relations', fontsize=50, y=1.08)
    axs = np.array(axs)

    for i in range(len(t)):
        newt = pd.concat([t[i]['airmar']['distance'][1:],
                              t[i]['ctd']['Salinity'][1:],
                              t[i]['ctd']['Temperature'][1:],
                              t[i]['gga']['CH4_ppm_adjusted'][1:],
                              t[i]['gga']['CO2_ppm_adjusted'][1:],
                              t[i]['op']['O2Concentration'][1:],
                              t[i]['nitrate']['0.00'][1:]],
                              axis=1)
        newt = newt.sort_values('distance')
        rt = newt.rolling(window=60)
        x = t[i]['airmar']['distance'][1:].apply(lambda m: round(m, -2))

        axs[0,i].set_title(labels[i], fontsize=40)

        for j in range(len(measures)):
            axs[j,0].set_ylabel(mlabels[j], fontsize=30)
            axs[j,i].plot(t[i]['airmar']['distance'],
                          t[i][instruments[j]][measures[j]],
                          'bo',
                          alpha=0.2,
                          label='Raw Data')

            # axs[j,i].plot(x,
            #               t[i][instruments[j]][measures[j]][1:],
            #               'ro',
            #               label='Rounded Aggregate')

            axs[j,i].plot(rt['distance'].mean(), rt[measures[j]].mean(), 'r-', label='Average')
            axs[j,i].axvline(x=0, c='k')

    fig.tight_layout()

def plot_distance_master_2(ta, tb, tc, td, te):
    t = [ta, tb, tc, td, te]
    labels = ['A', 'B', 'C', 'D', 'E']
    instruments = ['ctd', 'ctd', 'gga', 'gga', 'nitrate', 'op']
    measures = ['Salinity', 'Temperature', 'CH4_ppm_adjusted', 'CO2_ppm_adjusted', '0.00', 'O2Concentration']
    mlabels=['Salinity', 'Temperature', 'CH4', 'CO2', 'Nitrate', 'O2']

    fig, axs = plt.subplots(5,6, figsize=(40,40), sharex='col', sharey='col')
    fig.suptitle('Distance Relations', fontsize=50, y=1.08)
    axs = np.array(axs)

    for i in range(len(t)):
        newt = pd.concat([t[i]['airmar']['distance'][1:],
                              t[i]['ctd']['Salinity'][1:],
                              t[i]['ctd']['Temperature'][1:],
                              t[i]['gga']['CH4_ppm_adjusted'][1:],
                              t[i]['gga']['CO2_ppm_adjusted'][1:],
                              t[i]['op']['O2Concentration'][1:],
                              t[i]['nitrate']['0.00'][1:]],
                              axis=1)
        newt = newt.sort_values('distance')
        rt = newt.rolling(window=60)
        x = t[i]['airmar']['distance'][1:].apply(lambda m: round(m, -2))

        axs[i,0].set_ylabel(labels[i], fontsize=40)

        for j in range(len(measures)):
            axs[0,j].set_title(mlabels[j], fontsize=30)
            axs[i,j].plot(t[i]['airmar']['distance'],
                          t[i][instruments[j]][measures[j]],
                          'bo',
                          alpha=0.2,
                          label='Raw Data')

            # axs[j,i].plot(x,
            #               t[i][instruments[j]][measures[j]][1:],
            #               'ro',
            #               label='Rounded Aggregate')

            axs[i,j].plot(rt['distance'].mean(), rt[measures[j]].mean(), 'r-', label='Average')
            axs[i,j].axvline(x=0, c='k')

    fig.tight_layout()

   