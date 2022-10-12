import os
import argparse
from datetime import datetime
import time
# import msvcrt
# import keyboard


def deltatime(mytime, stime):
	diff_seconds = (mytime - stime).total_seconds()
	return diff_seconds


stime = None
# a_time = None
time_list = []
atime_list = []
count = 0

def writ_txt(t_l,a_l):
	global count
	with open('/media/wei/Seagate/Nemo/timelist.txt','a') as ff:
		ff.write(str(stime))
		ff.write('\n')
		ff.write(str(t_l))
		ff.write('\n')
		ff.write(str(a_l))
		ff.write('\n')
	with open('/home/wei/Downloads/timelist.txt','a') as ff:
		ff.write(str(stime))
		ff.write('\n')
		ff.write(str(t_l))
		ff.write('\n')
		ff.write(str(a_l))
		ff.write('\n')


def start_recording():
	global stime
	# global a_time
	global time_list
	global atime_list
	time_list = []
	atime_list = []
	os.system('curl http://3duniversum-iphonexs.local:8080/start')
	os.system('python3 /home/wei/Downloads/obsRemote.py start ')
	stime = datetime.now()
	atime_list.append(time.time())
	time_list.append(0)


def stop_recording():
	global time_list
	global atime_list
	os.system('curl http://3duniversum-iphonexs.local:8080/stop')
	os.system('python3 /home/wei/Downloads/obsRemote.py stop ')
	# splitting video
	# os.system('python /home/wei/Downloads/try/split.py -n dasdfas -t [dafasd]')
	mytime = datetime.now()
	dt = deltatime(mytime, stime)
	atime_list.append(time.time())
	time_list.append(dt)
	print(time_list)

	writ_txt(time_list,atime_list)
	time_list = []
	atime_list = []




def start_next():
	global  stime
	global time_list
	global atime_list
	mytime = datetime.now()
	dt = deltatime(mytime, stime)
	atime_list.append(time.time())
	time_list.append(dt)
















# 'curl http://3duniversum-iphonexs.local:8080/stop'

# def get_args():
# 	parser = argparse.ArgumentParser(description="this is the video seperator",
# 	                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# 	parser.add_argument("--time", "-t", type=list, default=[0,60],
# 	                    help="the time need to be separated")
# 	parser.add_argument("--name", "-n", type=str, default="try",
# 	                    help="clip names")
#
# 	args = parser.parse_args()
# 	return args
#
#



# while True:
# 	if getButton:
# 		print("time start")
# 		stime = datetime.now()
# 		a_time = time.time()
# 		atime_list.append(a_time)
# 		print(stime)
# 		print(a_time)
# 		break
#
# time.sleep(2)
#
# while len(time_list) < 4:
# 	if keyboard.is_pressed('q'):
# 		mytime = datetime.now()
# 		dt = deltatime(mytime, stime)
# 		a_time = time.time()
# 		atime_list.append(a_time)
# 		print("time:%d" % dt)
# 		time_list.append(dt)
# 		time.sleep(2)
# 	if keyboard.is_pressed('p'):
# 		break