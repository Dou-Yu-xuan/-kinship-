import os
import pickle
# from moviepy.editor import VideoFileClip

vi_path = 'vi-4-20'
vi_list = [it for it in os.listdir(vi_path) if it.endswith('.mp4')]
vi_list = sorted(vi_list)
lth = len(vi_list)
len_dict = {}
for i in range(lth):
	vi_loc = os.path.join(vi_path,vi_list[i])
	clip = VideoFileClip(vi_loc)
	print( clip.duration )
	len_dict[vi_list[i]] = clip.duration

# print(len_dict)

# def save_txt(name, len_dict):
# 	with open(name,'w') as ff:
# 		for items in len_dict:
# 			lines = '{}:{}\n'.format(items,len_dict[items])
# 			ff.writelines(lines)
#
# name = '{}.txt'.format(vi_path)
#
# save_txt(name,len_dict)

##############
name = '{}.pkl'.format(vi_path)
with open(name,'wr') as ff:
	pickle.dump(len_dict,ff)


# with open(name,'r') as ff:
# 	mmp = pickle.load(ff)
#
# print(mmp['2019-04-20-12-41-03.mp4'])