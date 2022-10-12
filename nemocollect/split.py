import os
import argparse

def get_args():
	parser = argparse.ArgumentParser(description="this is the video seperator",
	                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	# 0, 9.186185, 29.11801, 42.944178, 63.927365, 100
	parser.add_argument("--time", "-t", type=str, default='',
	                    help="the time need to be separated")
	parser.add_argument("--name", "-n", type=str, default='',
	                    help="clip names")
	parser.add_argument("--group", "-g", type=str, default='',
	                    help="group-language-number: 1000-en-1")

	args = parser.parse_args()
	return args

# def write_list(vi_list):
# 	with open('vi.txt','w') as ff:
# 		for vi in vi_list:
# 			ff.write(vi+'\n'
#
# def read_list():
# 	with open('vi.txt','r') as ff:
# 		vi_list = ff.readlines()
# 	r_list = [vi.replace('\n','') for vi in vi_list]
# 	return  r_list


vi_path = '/media/wei/Seagate/Nemo/recording'
litime = '/media/wei/Seagate/Nemo/timelist.txt'
list_presentation = '/media/wei/Seagate/Nemo/presentations_orders/'
seg_path = '/media/wei/Seagate/Nemo/sep'

def get_time(litime):
	with open(litime,'r') as ff:
		list_ti = ff.readlines()
	t_list = list_ti[-2].replace('\n','')
	return t_list,list_ti[-3]


def new_vi(vi_path):
	vi_list = [vi for vi in os.listdir(vi_path) if vi.endswith('.mp4')]
	sortlist= sorted(vi_list)
	return  sortlist[-1]

def get_spln(split_name_list,num):
	num = int(num)
	spli = os.path.join(list_presentation,split_name_list)
	with open(spli,'r') as ff:
		spnam = ff.readlines()
	spnam = [sn for sn in spnam if not sn=='\n']
	spli_line = spnam[num-1]
	outlis = spli_line.split(' ')
	return outlis[2][:-1],outlis[3]

def main():
	args = get_args()
	if not args.time == '':
		t_list = args.time
		lis_t = t_list.split(',')
	else:
		t_list,ck_nam = get_time(litime)
		lis_t = t_list[1:-1].split(',')

	if not args.name == '':
		vi_name = args.name
	else:
		vi_name = new_vi(vi_path)

	print(vi_name)
	print(ck_nam)

	if not args.group == '':
		gp_f = args.group
		gp_n,lan,num = gp_f.split('-')
		split_name_list = 'nemo_presentation_{}_order_{}.txt'.format(gp_n,lan)
		name_one, name_two = get_spln(split_name_list,num)
		if name_one.endswith('.avi'):
			name_one =name_one.replace('.avi','.mp4')
		if name_two.endswith('.avi'):
			name_two =name_two.replace('.avi','.mp4')
		print(name_one)
		print(name_two)

	assert name_one[:4]==gp_n
	assert name_two[:4]==gp_n
	seg_group_path = os.path.join(seg_path,gp_n)
	if not os.path.exists(seg_group_path):
		os.makedirs(seg_group_path)

	input_name = os.path.join(vi_path,vi_name)

	spli_nam1 = os.path.join(seg_group_path,name_one)
	spli_nam2 = os.path.join(seg_group_path,name_two)
	vi_command = "/usr/bin/ffmpeg -i {} -ss {} -to {} {}".format(input_name,
	                            float(lis_t[1]), float(lis_t[2]),spli_nam1)
	os.system(vi_command)

	vi_command2 = "/usr/bin/ffmpeg -i {} -ss {} -to {} {}".format(input_name,
	                            float(lis_t[3]), float(lis_t[4]),spli_nam2)
	os.system(vi_command2)



	# # assert  len(args.time) == 2
	# time_list = args.time.split(',')
	# vi_list = read_list()
	# ori_list = vi_list
	# name1 = args.name
	# name2 = args.name
	# for vi in os.listdir(root_path):
	# 	if not vi.endswith('.mp4'):
	# 		continue
	# 	# vi_list.append(vi)
	# 	# name = os.path.join(vi)
	# 	if not vi in ori_list:
	# 		im_path = os.path.join(root_path,vi)
	# 		vi_command = "/usr/bin/ffmpeg -i {} -ss {} -to {} sep/{}".format(im_path,float(time_list[1]),float(time_list[2]),args.name)
	# 		os.system(vi_command)
	# 		vi_command_2 = "/usr/bin/ffmpeg -i {} -ss {} -to {} sep/{}_1".format(im_path,float(time_list[3]),float(time_list[4]),args.name)
	# 		os.system(vi_command_2)
	# 		vi_list.append(vi)
	# 		write_list(vi_list)



if __name__ == '__main__':
	main()

