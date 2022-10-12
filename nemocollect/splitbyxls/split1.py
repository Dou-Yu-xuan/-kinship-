import os
import pandas as pd
import datetime
import time

def transtime(x):
	tlist =x.split(':')
	tsecon = sum(x * float(t) for x, t in zip([3600, 60, 1, 1./60], tlist))
	return tsecon

def split_vi(xname,x_path,vi_path,sep_path):
	# xname = '2019-04-20-16-49-33.xlsx'
	viname = vi_path+ xname.split('.')[0]+'.mp4'
	x_dir = os.path.join(x_path,xname)
	a = pd.read_excel(x_dir)
	sep_dir = os.path.join(sep_path,xname.split('.')[0])


	if not os.path.exists(sep_dir):
		os.makedirs(sep_dir)

	for i in range(17):

		tstop = transtime(a['End'][i])
		if not (i == 1 or i == 3):
			tstar = tstop - 1
		else:
			tstar = transtime(a['Start'][i])
			# print('start:%f'%tstar)
			# print('stop:%f \n'%tstop)
		#
		tstop = tstop+0.8
		tstar = tstar-0.6
		print('start:%f'%tstar)
		print('stop:%f \n'%tstop)

		if i == 1 or i ==3:
			comm = '/usr/bin/ffmpeg -i {} -ss {} -to {} {}/{}_D{}.mp4'.format(viname,
			                                                                    tstar, tstop, sep_dir, i, a['Answers'][i])
		elif i in range(8,11):
			comm = '/usr/bin/ffmpeg -i {} -ss {} -to {} {}/{}_{}_{}.mp4'.format(viname,
																		  tstar,tstop, sep_dir,i,a['Answers'][i],a['Hand'][8])
		elif i in range(11,14):
			comm = '/usr/bin/ffmpeg -i {} -ss {} -to {} {}/{}_{}_{}.mp4'.format(viname,
																		  tstar, tstop, sep_dir,i, a['Answers'][i],a['Hand'][11])
		elif i in range(14,17):
			comm = '/usr/bin/ffmpeg -i {} -ss {} -to {} {}/{}_{}_{}.mp4'.format(viname,
																		  tstar, tstop,sep_dir, i, a['Answers'][i], a['Hand'][14])
		# elif i==1 or i ==3:
		# 	comm = '/usr/bin/ffmpeg -i {} -ss {} -to {} {}_{}.mp4'.format(viname,
		# 	                                                                 tstar, tstop, i, a['Answers'][i])
		else:
			comm = '/usr/bin/ffmpeg -i {} -ss {} -to {} {}/{}.mp4'.format(viname,tstar, tstop,sep_dir, i)

		os.system(comm)

if __name__ == '__main__':

	for xname in os.listdir('4-20'):
		print(xname)
		split_vi(xname, '4-20','', 'vi-4-20')
