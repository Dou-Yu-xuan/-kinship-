import pandas as pd
import os
import datetime
import time
import math


def cvtime(time):
	a = time.split(':')[1]
	b = int(time.split(':')[-1].split('.')[0])
	c = time.split(':')[-1].split('.')[-1]
	c = int(float('0.'+c)*60)
	newt = '00:%s:%02d:%02d'%(a,b,c)
	return newt

def contime2(time):
	tsplit = time.split(':')
	sec = sum(float(t)*i for i,t in zip([60,1,1./60],tsplit))
	return sec+1.2



def change_excel(xpath):
	for xname in os.listdir(xpath):
		a = pd.read_excel(xpath + xname)
		savename = xname.split('.')[0] + '_new.xlsx'
		if math.isnan(a['End'][0]):
			for i in range(17):
				if i == 1 or i == 3:
					a['Start'][i] = '00:' + str(a['Start'][i])
					a['End'][i] = '00:' + str(a['End'][i])
					continue
				temt = str(a['Start'][i])
				sec = contime2(temt)
				addt = str(datetime.timedelta(seconds=sec))
				temt = cvtime(addt)
				a['Start'][i] = None
				a['End'][i] = temt
			a.to_excel(savename, index=False)

if __name__ == '__main__':
	xpath = '2/'
	change_excel(xpath)
