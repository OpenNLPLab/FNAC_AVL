import os
import random
train_data_path = '/home/notebook/data/personal/S9050086/vggsound/'

audio_path = f"{train_data_path}/audio/"
image_path = f"{train_data_path}/frames/"

# audio_path = f"{args.train_data_path}/"
# image_path = f"{args.train_data_path}/"

# List directory
audio_files = {fn.split('.wav')[0] for fn in os.listdir(audio_path) if fn.endswith('.wav')}
image_files = {fn.split('.png')[0] for fn in os.listdir(image_path) if fn.endswith('.png')}
avail_files = audio_files.intersection(image_files)
print(f"{len(avail_files)} available files")

subset = set(open(f"vggss_heard_test.txt").read().splitlines())
avail_files = avail_files.intersection(subset)
print(f"{len(avail_files)} valid subset files")

print(len(image_files), len(audio_files))
avail_files = list(avail_files)

print(len(avail_files))
count = 1

while len(avail_files) < 144000:
    cur = random.choice(list(image_files))
    count += 1
    
    if cur not in avail_files and cur in audio_files:
        avail_files.append(cur)
        print(count, cur)
        
    
print(f"{len(avail_files)} valid subset files")
# with open('vggss_144k_2.txt', 'w') as f:
    # for line in avail_files:
        # f.write(f"{line}\n", flush=True)

