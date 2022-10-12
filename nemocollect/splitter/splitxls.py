import os
import pandas as pd
import datetime
import time

xname = '2019-04-20-13-52-09.xlsx'
viname = xname.split('.')[0]+'.mp4'
a = pd.read_excel(xname)

def transtime(x):
	tlist =x.split(':')
	tsecon = sum(x * float(t) for x, t in zip([3600, 60, 1, 1./60], tlist))
	return tsecon



for i in range(17):

	tstop = transtime(a['End'][i])
	if not (i ==1 or i ==3):
		tstar = tstop - 1
	else:
		tstar = transtime(a['Start'][i])
		print('start:%f'%tstar)
		print('stop:%f \n'%tstop)

	if i in range(8,11):
		comm = '/usr/bin/ffmpeg -i {} -ss {} -to {} {}_{}_{}.mp4'.format(viname,
																	  tstar,tstop, i,a['Answers'][i],a['Hand'][8])
	elif i in range(11,14):
		comm = '/usr/bin/ffmpeg -i {} -ss {} -to {} {}_{}_{}.mp4'.format(viname,
																	  tstar, tstop, i, a['Answers'][i],a['Hand'][11])
	elif i in range(14,17):
		comm = '/usr/bin/ffmpeg -i {} -ss {} -to {} {}_{}_{}.mp4'.format(viname,
																	  tstar, tstop, i, a['Answers'][i], a['Hand'][14])
	# elif i==1 or i ==3:
	# 	comm = '/usr/bin/ffmpeg -i {} -ss {} -to {} {}_{}.mp4'.format(viname,
	# 	                                                                 tstar, tstop, i, a['Answers'][i])
	else:
		comm = '/usr/bin/ffmpeg -i {} -ss {} -to {} {}.mp4'.format(viname,tstar, tstop, i)

	os.system(comm)