import os
import pandas as pd
import datetime
import time

xname = '2019-04-20-16-49-33.xlsx'
viname = xname.split('.')[0]+'.mp4'
a = pd.read_excel(xname)

def transtime(x):
	tlist =x.split(':')
	tsecon = sum(x * float(t) for x, t in zip([60, 1], tlist))
	return tsecon

if not os.path.exists(xname.split('.')[0]):
	os.makedirs(xname.split('.')[0])

for i in range(17):

	tstar = transtime(a['Start'][i])
	if not (i ==1 or i ==3):
		tstop = tstar + 1
	else:
		tstop = transtime(a['End'][i])
		print('start:%f'%tstar)
		print('stop:%f \n'%tstop)
	#
	if i in range(8,11):
		comm = '/usr/bin/ffmpeg -i {} -ss {} -to {} {}/{}_{}_{}.mp4'.format(viname,
																	  tstar,tstop, xname.split('.')[0],i,a['Answer'][i],a['Hand'][8])
	elif i in range(11,14):
		comm = '/usr/bin/ffmpeg -i {} -ss {} -to {} {}/{}_{}_{}.mp4'.format(viname,
																	  tstar, tstop, xname.split('.')[0],i, a['Answer'][i],a['Hand'][11])
	elif i in range(14,17):
		comm = '/usr/bin/ffmpeg -i {} -ss {} -to {} {}/{}_{}_{}.mp4'.format(viname,
																	  tstar, tstop,xname.split('.')[0], i, a['Answer'][i], a['Hand'][14])
	# elif i==1 or i ==3:
	# 	comm = '/usr/bin/ffmpeg -i {} -ss {} -to {} {}_{}.mp4'.format(viname,
	# 	                                                                 tstar, tstop, i, a['Answers'][i])
	else:
		comm = '/usr/bin/ffmpeg -i {} -ss {} -to {} {}/{}.mp4'.format(viname,tstar, tstop,xname.split('.')[0], i)

	os.system(comm)