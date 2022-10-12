import os
import pandas as pd
from tqdm import tqdm
import datetime
import time
import math
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

	# print(sep_dir)
	# return

	if not os.path.exists(sep_dir):
		os.makedirs(sep_dir)

	row_nums = len(a['End'])
	for i in range(row_nums):
		temend = a['End'][i]
		if not isinstance(temend,str) or temend =='none':
			print(sep_dir)
			continue
		tstop = transtime(a['End'][i])
		if not (i == 1 or i == 3):
			tstar = tstop - 1
		else:
			tstar = transtime(a['Start'][i])
			# print('start:%f'%tstar)
			# print('stop:%f \n'%tstop)
		#
		tstop = tstop
		tstar = tstar
		# print('start:%f'%tstar)
		# print('stop:%f \n'%tstop)

		if i == 1 or i ==3:
			comm = 'ffmpeg  -nostats -loglevel 0 -i  {} -ss {} -to {} {}/{}_D{}.mp4'.format(viname,
			                                                                    tstar, tstop, sep_dir, i, a['Answers'][i])
		elif i in range(8,11):
			comm = 'ffmpeg -nostats -loglevel 0 -i {} -ss {} -to {} {}/{}_{}_{}.mp4'.format(viname,
																		  tstar,tstop, sep_dir,i,a['Answers'][i],a['Hand'][8])
		elif i in range(11,14):
			comm = 'ffmpeg -nostats -loglevel 0 -i {} -ss {} -to {} {}/{}_{}_{}.mp4'.format(viname,
																		  tstar, tstop, sep_dir,i, a['Answers'][i],a['Hand'][11])
		elif i in range(14,17):
			comm = 'ffmpeg -nostats -loglevel 0 -i {} -ss {} -to {} {}/{}_{}_{}.mp4'.format(viname,
																		  tstar, tstop,sep_dir, i, a['Answers'][i], a['Hand'][14])
		# elif i==1 or i ==3:
		# 	comm = '/usr/bin/ffmpeg -i {} -ss {} -to {} {}_{}.mp4'.format(viname,
		# 	                                                                 tstar, tstop, i, a['Answers'][i])
		else:
			comm = 'ffmpeg -nostats -loglevel 0 -i {} -ss {} -to {} {}/{}.mp4'.format(viname,tstar, tstop,sep_dir, i)

		os.system(comm)

if __name__ == '__main__':
	Time_ls = ['4-29','4-30','5-1','5-2','5-3']
	for i in range(len(Time_ls)):
		TIME = Time_ls[i]
		x_path = 'sp_time/Time-'+TIME
		# i = 0
		for xname in tqdm(sorted(os.listdir(x_path))):
			# print(xname)
			# i +=1
			# if i< 17:
			# 	continue
			vi_path = 'webcam/'+TIME+'/recording/'
			# print(vi_path)
			sep_path = 'sp_video/vi-'+TIME
			split_vi(xname, x_path,vi_path,sep_path)


