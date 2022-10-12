import os
import pickle
import json
from datetime import datetime
import time
import pandas as pd

def read_dict(name):
	with open(name,'rb') as ff:
		return  pickle.load(ff)

def count_num(files):
	num = (len(os.listdir(files))-1)/3
	return num

def load_ts(ts):
	# load time stamp
	tss = os.path.join('timelist',ts)
	with open(tss,'r') as ff:
		return ff.readlines()

def extract_ts(ts_list,idx):
	ts_time = ts_list[3*idx]
	ts_ts = ts_list[3*idx+2]
	ts_ts = ts_ts.replace('\n',''). \
		replace('[', '').replace(']', '').split(',')
	ts_ts = [float(it) for it in ts_ts]
	return ts_time, ts_ts

def load_json(file):
	with open(file) as ff:
		data = json.load(ff)
	return data

def transtime(x):
	tlist =x.split(':')
	tsecon = sum(x * float(t) for x, t in zip([3600, 60, 1, 1./60], tlist))
	return tsecon

class FETCH(object):
	# def __init__(self):

	def get_num(self,a_json,ori_time,start,stop):
		self.lenth = len(a_json) - 1
		self.start_t = ori_time + start
		self.stop_t = ori_time + stop

	def num(self):
		pass



def fetch_num(a_json,ori_time,start,stop):

	# def get_num(ttime,lenth):
	# 	for it in range(lenth-1):
	# 		if ttime
	# def


	lenth = len(a_json)-1
	start_t = ori_time +start
	stop_t = ori_time+stop
	n_start =-1
	n_stop = -1
	comp_t = a_json[1]['epochTime']
	# def get_it(json):


	if comp_t<start_t:
		# all is the same
		for it in range(lenth):
			if a_json[it]['epochTime']<=start_t and a_json[it+1]['epochTime']>start_t:
				n_start = it-1
				break
		for it in range(lenth):
			if a_json[it]['epochTime']<=stop_t and a_json[it+1]['epochTime']>stop_t:
				n_stop = it-1
				break
		if n_start == n_stop:
			return -1, -1
		return n_start, n_stop

	elif comp_t<stop_t:
		# [comp_t,stop_t]
		n_start = 0
		for it in range(lenth):
			if a_json[it]['epochTime']<=stop_t and a_json[it+1]['epochTime']>stop_t:
				n_stop = it-1
		if n_start == n_stop:
			return -1, -1
		return n_start, n_stop

	else:
		n_start, n_stop = -1, -1
		return n_start, n_stop





def write_dict(txt_path,name,num_dict):
	if not os.path.exists(txt_path):
		os.makedirs(txt_path)
	txt_name = name+'.txt'
	txt_name = os.path.join(txt_path,txt_name)
	with open(txt_name,'w') as ff:
		for it in num_dict:
			line = '{}\t{}\t{}\n'.format(it,num_dict[it][0],num_dict[it][1])
			ff.write(line)


def fetch_num_without_json(tstar, tstop, vi_len, num_ip):

	if tstop > 0:
		num_stop = int(tstop / vi_len * num_ip)
		if tstar > 0:
			num_start = int(tstar / vi_len * num_ip)
		else:
			num_start = 0
	else:
		num_start = -1
		num_stop = -1

	return num_start,num_stop

def r_vi_name(vi_path,endwith):
	vi_list = [it for it in os.listdir(vi_path) if it.endswith(endwith)]
	vi_list = sorted(vi_list)
	return vi_list

if __name__ == '__main__':

	div_frames = []
	# xls path, video path and iphone path
	DATE = '5-3'
	xls_path = '/media/wei/Data/iphone/xls/Time-{}'.format(DATE)
	vi_path = '/media/wei/Doc/Nemo/{}/recording'.format(DATE)
	ip_path = DATE
	dict_path = '/media/wei/Data/iphone/vi_lenth/vi-{}.pkl'.format(DATE)
	spli_path = '/media/wei/Data/iphone/sp_iphone/{}'.format(DATE)
	# ts_path = 'timelist-4-20.txt'


	xls_list = r_vi_name(xls_path,'.xlsx')
	vi_list = r_vi_name(vi_path,'.mp4')
	ip_list = sorted(os.listdir(ip_path))
	# ts_list = load_ts(ts_path)

	# a,b = extract_ts(ts_list,0)


	lth = len(xls_list)
	# dict_name = '{}.pkl'.format(vi_path)
	len_dict = read_dict(dict_path)


	#########


	#########
	for item in range(lth):
		print(xls_list[item])
		# load xls,
		xls_loc = os.path.join(xls_path,xls_list[item])
		xls = pd.read_excel(xls_loc)

		# open iphone dir
		ip_loc = os.path.join(ip_path,ip_list[item])

		# if there is timestamps.json in it
		spec_json = os.path.join(ip_loc,'timestamps.json')
		ifexit = os.path.exists(spec_json)
		# a,b = extract_ts(ts_list,i)
		if ifexit:
			json_t = load_json(spec_json)
			num_dict = {}
			ori_time = float(ip_list[item].split(' ')[0])
			for i in range(17):
				try:

					tstop = transtime(xls['End'][i])
				except:
					tstar = -1
					tstop = -1
					if i == 1 or i == 3:
						try:
							comm = '{}_D{}'.format(i, xls['Answers'][i])
						except:
							comm = '{}'.format(i)
						num_dict[comm] = [-1, -1]
					elif i in range(8, 11):
						try:
							comm = '{}_{}_{}'.format(i, xls['Answers'][i], xls['Hand'][8])
						except:
							comm = '{}'.format(i)
						num_dict[comm] = [-1, -1]
					elif i in range(11, 14):
						try:
							comm = '{}_{}_{}'.format(i, xls['Answers'][i], xls['Hand'][11])
						except:
							comm = '{}'.format(i)
						num_dict[comm] = [-1, -1]
					elif i in range(14, 17):
						try:
							comm = '{}_{}_{}'.format(i, xls['Answers'][i], xls['Hand'][14])
						except:
							comm = '{}'.format(i)
						num_dict[comm] = [-1, -1]
					else:
						comm = '{}'.format(i)
						num_dict[comm] = [-1, -1]
					continue
				if not (i == 1 or i == 3):
					tstar = tstop - 1
				else:
					tstar = transtime(xls['Start'][i])
					# print('start:%f'%tstar)
					# print('stop:%f \n'%tstop)
				#
				# if i == 0:
				# 	pass
				#
				# else:
				#
				tstop = tstop+0.8
				tstar = tstar-0.6
				num_start,num_stop = fetch_num(json_t,ori_time,tstar,tstop)

				if i == 1 or i ==3:
					comm = '{}_D{}'.format(i, xls['Answers'][i])
					num_dict[comm]= [num_start,num_stop]
				elif i in range(8,11):
					comm = '{}_{}_{}'.format(i,xls['Answers'][i],xls['Hand'][8])
					num_dict[comm]= [num_start,num_stop]
				elif i in range(11,14):
					comm = '{}_{}_{}'.format(i, xls['Answers'][i],xls['Hand'][11])
					num_dict[comm]= [num_start,num_stop]
				elif i in range(14,17):
					comm = '{}_{}_{}'.format(i, xls['Answers'][i], xls['Hand'][14])
					num_dict[comm]= [num_start,num_stop]
				else:
					comm = '{}'.format(i)
					num_dict[comm] = [num_start, num_stop]
			ip_txt = spli_path+'_txt'
			write_dict(ip_txt,ip_list[item],num_dict)

		else:
			num_dict = {}
			vi_len = len_dict[vi_list[item]]-3
			for i in range(17):

				try:
					tstop = transtime(xls['End'][i])
				except:
					tstar = -1
					tstop = -1
					if i == 1 or i == 3:
						comm = '{}_D{}'.format(i, xls['Answers'][i])
						num_dict[comm] = [-1, -1]
					elif i in range(8, 11):
						try:
							comm = '{}_{}_{}'.format(i, xls['Answers'][i], xls['Hand'][8])
						except:
							comm = '{}'.format(i)
						num_dict[comm] = [-1, -1]
					elif i in range(11, 14):
						try:
							comm = '{}_{}_{}'.format(i, xls['Answers'][i], xls['Hand'][11])
						except:
							comm = '{}'.format(i)
						num_dict[comm] = [-1, -1]
					elif i in range(14, 17):
						try:
							comm = '{}_{}_{}'.format(i, xls['Answers'][i], xls['Hand'][14])
						except:
							comm = '{}'.format(i)
						num_dict[comm] = [-1, -1]
					else:
						comm = '{}'.format(i)
						num_dict[comm] = [-1, -1]
					continue

				if not (i == 1 or i == 3):
					tstar = tstop - 1
				else:
					tstar = transtime(xls['Start'][i])

				tstop = tstop+0.8
				tstar = tstar-0.6
				tstar -=7
				tstop  -=7
				num_ip = int((len(os.listdir(ip_loc))-1)/3)
				# print(num_ip)
				# print(vi_len)

				num_start, num_stop = fetch_num_without_json(tstar,tstop,vi_len,num_ip)

				if i == 1 or i ==3:
					comm = '{}_D{}'.format(i, xls['Answers'][i])
					num_dict[comm]= [num_start,num_stop]
				elif i in range(8,11):
					comm = '{}_{}_{}'.format(i,xls['Answers'][i],xls['Hand'][8])
					num_dict[comm]= [num_start,num_stop]
				elif i in range(11,14):
					comm = '{}_{}_{}'.format(i, xls['Answers'][i],xls['Hand'][11])
					num_dict[comm]= [num_start,num_stop]
				elif i in range(14,17):
					comm = '{}_{}_{}'.format(i, xls['Answers'][i], xls['Hand'][14])
					num_dict[comm]= [num_start,num_stop]
				else:
					comm = '{}'.format(i)
					num_dict[comm] = [num_start, num_stop]

			ip_txt = spli_path+'_txt'
			write_dict(ip_txt,ip_list[item],num_dict)








			# json_t[0]['epochTime'] - float(ip_list[0].split(' ')[0])
			# print(json_t)
			# print()




	# print(ts_list)
	# print(b[0])
	# print(ip_list)
	# print(ip_list[0].split(' ')[0])
	#
	# print(time.time())
	# print(datetime.now())

