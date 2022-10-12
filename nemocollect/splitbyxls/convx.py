import pandas as pd
import os
import datetime
import math
import time

def cvtime(time):
	a = time.split(':')[1]
	b = int(time.split(':')[-1].split('.')[0])
	c = time.split(':')[-1].split('.')[-1]
	c = int(float('0.'+c)*60)
	newt = '00:%s:%02d:%02d'%(a,b,c)
	return newt

def cvtime1(time):
	a = time.split(':')[0]
	b = int(time.split(':')[-1].split('.')[0])
	c = time.split(':')[-1].split('.')[-1]
	c = int(float('0.'+c)*60)
	newt = '00:%s:%02d:%02d'%(a,b,c)
	return newt

def toseconds(time):
	tlist = time.split(':')
	sec = sum(float(x)*i for x,i in zip(tlist,[60,1]))
	return sec+1.2

def change_excel(path):
	for item in os.listdir(path):
		print(item)
		nw_item = item.split('.')[0]+'_new'+'.xlsx'
		toconv = pd.read_excel(path+item)
		print(toconv)
		check = toconv['Start'][0]
	#     print(toconv)
		if isinstance(check,str):
			for i in range(17):
				if i ==1 or i ==3:
					toconv['Start'][i] = cvtime1(toconv['Start'][i])
					toconv['End'][i] = cvtime1(toconv['End'][i])
					continue
				temt = toconv['Start'][i]
				sec = toseconds(temt)
				addt = str(datetime.timedelta(seconds=sec))
				cvt = cvtime(addt)
				toconv['Start'][i] = None
				toconv['End'][i] = cvt
			toconv.to_excel(nw_item,index = False)

if __name__ == '__main__':
	change_excel('1/')