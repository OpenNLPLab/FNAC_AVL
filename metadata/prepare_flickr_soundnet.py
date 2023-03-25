import os
import random
from pydub import AudioSegment
import shutil
# import subprocess


file_list = open(f"flickr_144k.txt").read().splitlines()
print(len(file_list))

for index, name in enumerate(file_list):
    print(index)
    num_path = list(name[-3:])#.split()
    mp3_dir = os.path.join('/home/notebook/data/personal/S9050086/mp3/videos/',num_path[0], num_path[1], num_path[2], name+'.mp4.mp3')
    if os.path.isfile(mp3_dir):
        sound = AudioSegment.from_mp3(mp3_dir)
        # subprocess.call(['ffmpeg', '-i', mp3_dir,
        #            'test.wav'])
    else:
        num_path = list(name[-8:])#.split()
        mp3_dir = os.path.join('/home/notebook/data/personal/S9050086/mp3/videos2/',num_path[0], num_path[1], num_path[2],num_path[3], num_path[4], num_path[5],\
                                                                                               num_path[6], num_path[7], name+'.mp4.mp3')
        sound = AudioSegment.from_mp3(mp3_dir)
    

    frame_dir = os.path.join('/home/notebook/data/personal/S9050086/frames/videos/',num_path[0], num_path[1], num_path[2], name+'.mp4')
    if os.path.isdir(frame_dir):
        frames = os.listdir(frame_dir)
    else:
        num_path = list(name[-8:])#.split()
        frame_dir = os.path.join('/home/notebook/data/personal/S9050086/frames/videos2/',num_path[0], num_path[1], num_path[2],num_path[3], num_path[4], num_path[5],\
                                                                                               num_path[6], num_path[7],name+'.mp4')
        frames = os.listdir(frame_dir)





    print(mp3_dir)
    print(frame_dir)
    print(frames)


   
    frame_to_mv = random.choice(frames)
    audio_save_dir = os.path.join('/home/notebook/data/personal/S9050086/flickr/audio', name+'.wav')
    frame_save_dir = os.path.join('/home/notebook/data/personal/S9050086/flickr/frames', name+'.jpg')

    sound.export(audio_save_dir, format="wav")
    shutil.copy(os.path.join(frame_dir,frame_to_mv), frame_save_dir)
    


