import os
import pandas as pd
from tqdm import tqdm
import datetime
import time
import math
import pickle


sep_names ={'4-20':['001_1_nl','002_1_en','002_2_en','003_1_nl','003_2_nl','004_1_en','004_2_en','005_1_nl','006_1_en','006_2_en','007_1_en',
					'007_2_en','008_1_en','008_2_en','009_1_en','009_2_en','009_3_en'],
			'4-21':['010_1_nl','010_2_nl','011_1_nl','011_2_nl','012_1_nl','012_2_nl','012_3_nl','013_1_en','013_2_en','013_3_en','014_1_nl',
					'014_2_nl','015_1_nl','016_1_en','016_2_en','016_3_en','016_4_en','017_1_en','017_2_en','018_1_nl','018_2_nl','018_3_nl',
					'018_4_nl','019_1_nl','019_2_nl','019_3_nl','019_4_nl'],
			'4-22':['020_1_en','020_2_en','020_3_en','021_1_en','021_2_en','022_1_en','022_2_en','023_1_nl','023_2_nl','024_1_nl','025_1_nl',
					'025_2_nl'],
			'4-23':['026_1_nl','026_2_nl','027_1_nl','027_2_nl','028_1_en','028_2_en','028_3_en','028_4_en','028_5_en','029_1_en','029_2_en',
					'029_3_en','030_1_nl','030_2_nl','030_3_nl','030_4_nl','031_1_nl','031_2_nl','031_3_nl','031_4_nl','032_1_en','032_2_en'],
			'4-24':['033_1_en','033_2_en','033_3_en','034_1_en','034_2_en','034_3_en','034_4_en','035_1_nl','035_2_nl','035_3_nl','035_4_nl',
					'036_1_nl','036_2_nl','036_3_nl','037_1_en','037_2_en','037_3_en','037_4_en','038_1_en','038_2_en','039_1_en','039_2_en',
					'039_3_en'],
			'4-25':['040_1_nl','040_2_nl','041_1_en','041_2_en','042_1_nl','042_2_nl','042_3_nl','042_4_nl','042_5_nl','043_1_nl','043_2_nl',
					'043_3_nl','044_1_en','044_2_en','044_3_en','045_1_en','045_2_en','045_3_en','046_1_nl','046_2_nl'],
			'4-26':['047_1_nl','047_2_nl','047_3_nl','048_1_nl','048_2_nl','049_1_nl','049_2_nl','049_3_nl','049_4_nl','049_5_nl','050_1_en',
					'051_1_nl','051_2_nl','051_3_nl','052_1_en','052_2_en','053_1_en','053_2_en','053_3_en','053_4_en','053_5_en','054_1_nl',
					'054_2_nl'],
			'4-28':['055_1_nl','055_2_nl','055_3_nl','055_4_nl','056_1_nl','056_2_nl','056_3_nl','056_4_nl','057_1_nl','057_2_nl','057_3_nl',
					'057_4_nl','057_5_nl','058_1_nl','058_2_nl','058_3_nl','058_4_nl','058_5_nl','059_1_nl','059_2_nl','059_3_nl','060_1_nl',
					'060_2_nl','060_3_nl','060_4_nl','060_5_nl','061_1_en','061_2_en','061_3_en','061_4_en'],
			'4-29':['062_1_nl','062_2_nl','062_3_nl','062_4_nl','063_1_nl','063_2_nl','063_3_nl','064_1_nl','064_2_nl','064_3_nl','065_1_nl',
					'065_2_nl','065_3_nl','066_1_nl','066_2_nl','066_3_nl','066_4_nl','066_5_nl','067_1_nl','067_2_nl','067_3_nl','067_4_nl',
					'067_5_nl'],
			'4-30':['068_1_en','068_2_en','068_3_en','069_1_nl','069_2_nl','069_3_nl','069_4_nl','070_1_nl','070_2_nl','070_3_nl',
					'070_4_nl','071_1_en','071_2_en','071_3_en','072_1_en','072_2_en','072_3_en','072_4_en','073_1_nl','073_2_nl','073_3_nl',
					'073_4_nl','073_5_nl','074_1_nl','074_2_nl','074_3_nl','074_4_nl','075_1_en','075_2_en','075_3_en'],
			'5-1': ['076_1_nl','076_2_nl','076_3_nl','076_4_nl','077_1_nl','077_2_nl','077_3_nl','077_4_nl','078_1_en','078_2_en','079_1_nl',
					'079_2_nl','079_3_nl','079_4_nl','080_1_nl','080_2_nl','080_3_nl','080_4_nl','080_5_nl','081_1_nl','081_2_nl','081_3_nl',
					'081_4_nl','081_5_nl','082_1_nl','082_2_nl','082_3_nl','082_4_nl','082_5_nl','083_1_nl','083_2_nl','083_3_nl','083_4_nl',
					'083_5_nl'],
			'5-2': ['084_1_en','084_2_en','085_1_nl','085_2_nl','085_3_nl','085_4_nl','085_5_nl','086_1_nl','086_2_nl','086_3_nl','086_4_nl',
					'086_5_nl','087_1_en','087_2_en','088_1_nl','088_2_nl','088_3_nl','088_4_nl','088_5_nl','089_1_nl','089_2_nl','089_3_nl',
					'089_4_nl','090_1_nl','090_2_nl','090_3_nl','090_4_nl','091_1_nl','091_2_nl','091_3_nl'],
			'5-3': ['092_1_nl','092_2_nl','092_3_nl','092_4_nl','092_5_nl','093_1_en','093_2_en','094_1_nl','094_2_nl','094_3_nl','095_1_en',
					'095_2_en','095_3_en','096_1_nl','096_2_nl','096_3_nl','096_4_nl','en','en','en','en']}

def transtime(x):
	tlist =x.split(':')
	tsecon = sum(x * float(t) for x, t in zip([3600, 60, 1, 1./60], tlist))
	return tsecon

def split_vi(xname,x_path,vi_path,sep_path,sep_nam):
	# xname = '2019-04-20-16-49-33.xlsx'
	viname = vi_path+ xname.split('.')[0]+'.mp4'
	x_dir = os.path.join(x_path,xname)
	a = pd.read_excel(x_dir)
	# sep_dir = os.path.join(sep_path,xname.split('.')[0])
	sep_dir = os.path.join(sep_path,sep_nam.split('.')[0])


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
		if i ==0 or i ==2:
			tstop = tstop+0.2
			tstar = tstar-0.6

		# print('start:%f'%tstar)
		# print('stop:%f \n'%tstop)
		######

		if i == 1 or i ==3:
			tstop = tstop + 0.2
			tstar = tstar - 0.6
			comm = 'ffmpeg  -nostats -loglevel 0 -i  {0} -ss {1} -to {2} {3}/{4}_{5:02d}_D{6}.mp4'.format(viname,
			                                                                    tstar, tstop, sep_dir,sep_nam, i, a['Answers'][i])
		elif i in range(8,11):
			comm = 'ffmpeg -nostats -loglevel 0 -i {0} -ss {1} -to {2} {3}/{4}_{5:02d}_{6}_{7}.mp4'.format(viname,
																		  tstar,tstop, sep_dir,sep_nam,i,a['Answers'][i],a['Hand'][8])
		elif i in range(11,14):
			comm = 'ffmpeg -nostats -loglevel 0 -i {0} -ss {1} -to {2} {3}/{4}_{5:02d}_{6}_{7}.mp4'.format(viname,
																		  tstar, tstop, sep_dir,sep_nam,i, a['Answers'][i],a['Hand'][11])
		elif i in range(14,17):
			comm = 'ffmpeg -nostats -loglevel 0 -i {0} -ss {1} -to {2} {3}/{4}_{5:02d}_{6}_{7}.mp4'.format(viname,
																		  tstar, tstop,sep_dir, sep_nam,i, a['Answers'][i], a['Hand'][14])
		# elif i==1 or i ==3:
		# 	comm = '/usr/bin/ffmpeg -i {} -ss {} -to {} {}_{}.mp4'.format(viname,
		# 	                                                                 tstar, tstop, i, a['Answers'][i])
		else:
			comm = 'ffmpeg -nostats -loglevel 0 -i {0} -ss {1} -to {2} {3}/{4}_{5:02d}.mp4'.format(viname,tstar, tstop,sep_dir, sep_nam,i)

		os.system(comm)

if __name__ == '__main__':
	Time_ls = ['4-20','4-21','4-22','4-23','4-24','4-25','4-26','4-28','4-29', '4-30', '5-1', '5-2', '5-3']
	# for i in range(len(Time_ls)):
	# 	TIME = Time_ls[i]
	# 	x_path = 'sp_time/Time-'+TIME
	# 	sep_name = sep_names[TIME]
	# 	for i,xname in enumerate(tqdm(sorted(os.listdir(x_path)))):
	# 		# print(xname)
	# 		# i +=1
	# 		# if i< 17:
	# 		# 	continue
	# 		sep_nam = sep_name[i]
	# 		vi_path = 'webcam/'+TIME+'/recording/'
	# 		# print(vi_path)
	# 		sep_path = 'sp_video/vi-'+TIME
	# 		split_vi(xname, x_path,vi_path,sep_path,sep_nam)
		#check
		# assert len(sep_name) ==len(os.listdir(x_path)),'wrong path is {},len {},len {}'.format(TIME,len(sep_name),len(os.listdir(x_path)))
	lan_dic = {}
	for i in range(len(Time_ls)):
		TIME = Time_ls[i]
		x_path = '/media/wei/Data/Nemo/data_and_label/sp_video/vi-'+TIME
		sep_name = sep_names[TIME]
		assert len(sep_name)==len(sorted(os.listdir(x_path))),"the fold {} has unmatched number".format(TIME)
		for i,xname in enumerate(sorted(os.listdir(x_path))):
			# print(xname)
			# i +=1
			# if i< 17:
			# 	continue
			lan_dic[xname] = sep_name[i][-2:]


	with open('video_language.pkl','wb') as ff:
		pickle.dump(lan_dic,ff)

	with open('video_language.pkl','rb') as wf:
		bb = pickle.load(wf)
	print(bb)



