# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 18:01:59 2020

@author: Saif
"""

#%% import necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

#%% define custom function
def st(x, t):
    if t == True:
        if x <= 70:
            stat = 'CO not achieved'
        else:
            stat = 'CO achieved'
    else:
        stat = 'N/A'
    return stat

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped """
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)


        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)


                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

#%% initiate variables and read file
C1, C2, C3, C4, C5, C6 = 'CO1', 'CO2', 'CO3', 'CO4', 'C5', 'C6'
df = pd.read_excel('test.xlsx', skiprow=1, header=1)
file = open('info.txt', 'r')
info = file.readlines()[0]
info = info[:-1]
file.close()

#seperate co-wise
co1_data = df.filter(like=C1)
co2_data = df.filter(like=C2)
co3_data = df.filter(like=C3)
co4_data = df.filter(like=C4)
co5_data = df.filter(like=C5)
co6_data = df.filter(like=C6)

#%% calculate total co-marks and array them up
co1_sum = co1_data.sum(axis=1, skipna=True).tolist()
co2_sum = co2_data.sum(axis=1, skipna=True).tolist()
co3_sum = co3_data.sum(axis=1, skipna=True).tolist()
co4_sum = co4_data.sum(axis=1, skipna=True).tolist()
co5_sum = co5_data.sum(axis=1, skipna=True).tolist()
co6_sum = co6_data.sum(axis=1, skipna=True).tolist()

co1_sum = np.array(co1_sum)
co2_sum = np.array(co2_sum)
co3_sum = np.array(co3_sum)
co4_sum = np.array(co4_sum)
co5_sum = np.array(co5_sum)
co6_sum = np.array(co6_sum)

#%% co achievement percentage studentwise
co_list = []
available_co = []
[tc1, tc2, tc3, tc4, tc5, tc6] = [False, False, False, False, False, False]
p = [tc1, tc2, tc3, tc4, tc5, tc6]

if co1_sum[0] != 0:
    co1_std = co1_sum[1:]/co1_sum[0]*100
    tc1 = True
    available_co.append(C1)
else:
     co1_std = np.zeros(len(df)-1)
if co2_sum[0] != 0:
    co2_std = co2_sum[1:]/co2_sum[0]*100
    tc2 = True
    available_co.append(C2)
else:
    co2_std = np.zeros(len(df)-1)
if co3_sum[0] != 0:
    co3_std = co3_sum[1:]/co3_sum[0]*100
    tc3 = True
    available_co.append(C3)
else:
    co3_std = np.zeros(len(df)-1)
if co4_sum[0] != 0:
    co4_std = co4_sum[1:]/co4_sum[0]*100
    tc4 = True
    available_co.append(C4)
else:
    co4_std = np.zeros(len(df)-1)
if co5_sum[0] != 0:
    co5_std = co5_sum[1:]/co5_sum[0]*100
    tc5 = True
    available_co.append(C5)
else:
    co5_std = np.zeros(len(df)-1)
if co6_sum[0] != 0:
    co6_std = co6_sum[1:]/co6_sum[0]*100
    tc6 = True
    available_co.append(C6)
else:
    co6_std = np.zeros(len(df)-1)

#%%  creat new dataframe for students assessments
name_list = df["Name of the Student"][1:]
co_wise_assessment = pd.DataFrame({'Name of the Student': name_list,
                                   C1:co1_std, C2:co2_std, C3:co3_std,
                                   C4:co4_std, C5:co5_std, C6:co6_std})

#%% calculate co achievement
co1_passed_std = len(co1_std[co1_std >= 60])
co1_passed_prcnt = (co1_passed_std)/len(co1_std)*100
co1_status = st(co1_passed_prcnt, tc1)

co2_passed_std = len(co2_std[co2_std >= 60])
co2_passed_prcnt = (co2_passed_std)/len(co2_std)*100
co2_status = st(co2_passed_prcnt, tc2)

co3_passed_std = len(co3_std[co3_std >= 60])
co3_passed_prcnt = (co3_passed_std)/len(co3_std)*100
co3_status = st(co3_passed_prcnt, tc3)

co4_passed_std = len(co4_std[co4_std >= 60])
co4_passed_prcnt = (co4_passed_std)/len(co4_std)*100
co4_status = st(co4_passed_prcnt, tc4)

co5_passed_std = len(co5_std[co5_std >= 60])
co5_passed_prcnt = (co5_passed_std)/len(co5_std)*100
co5_status = st(co5_passed_prcnt, tc5)

co6_passed_std = len(co6_std[co6_std >= 60])
co6_passed_prcnt = (co6_passed_std)/len(co6_std)*100
co6_status = st(co6_passed_prcnt, tc6)

co_list = [C1, C2, C3, C4, C5, C6]
co_std_list = [co1_passed_std, co2_passed_std, co3_passed_std, co4_passed_std,
               co5_passed_std, co6_passed_std]
co_prcnt_list = [co1_passed_prcnt, co2_passed_prcnt, co3_passed_prcnt,
                 co4_passed_prcnt, co5_passed_prcnt, co6_passed_prcnt]
co_stat_list = [co1_status, co2_status, co3_status, co4_status,
                co5_status, co6_status]

co_wise_achievement = pd.DataFrame({'CO no.': co_list,
                                    'No of Student with CO >= 60%':co_std_list,
                                    '% of CO acheivement':co_prcnt_list,
                                    'Status of CO':co_stat_list})

#%% write an excell file and store all data
co_wise_achievement.to_excel('subject CO achievement.xlsx')
co_wise_assessment.to_excel('student CO assessment.xlsx')


#%% creat graph fo co
data = [co_list, (str(info), [co_prcnt_list])]
                       
N = len(data[0])
theta = radar_factory(N, frame='polygon')

spoke_labels = data.pop(0)
title, case_data = data[0]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='radar'))
fig.subplots_adjust(top=0.85, bottom=0.05)

ax.set_rgrids([20, 40, 60, 80, 100])
ax.set_title(title,  position=(0.5, 1.1), ha='center')

for d in case_data:
    line = ax.plot(theta, d)
    ax.fill(theta, d,  alpha=0.35)
ax.set_varlabels(spoke_labels)

plt.savefig('graph.png')
plt.show()























