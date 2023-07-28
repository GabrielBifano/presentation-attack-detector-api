import os
import shutil
from PIL import Image
import random


## This file was used to adequate my dataset to the
## specifications of the method i use to load it on
## pytorch Dataset and DataLoader classes


dataPath = '< some path >'

def get_images(data_path):
  images = []
  for root, directories, files in os.walk(data_path):
    for file in files:
      if file.endswith(".jpg") or file.endswith(".png"):
        images.append(os.path.join(root, file))
  return images

# makes sure every image is a jpg
for img in get_images(dataPath):
  if 'png' in str(img):
    new_img = img.replace('png', 'jpg')
    im1 = Image.open(img)
    im1.save(new_img)
    os.remove(img)

# makes sure 'spoof' and 'live' dirs,
# exist inside every numbered dir
for dir in os.listdir(dataPath):
  if 'spoof' not in os.listdir(f'{dataPath}/{dir}'):
    os.mkdir(f'{dataPath}/{dir}/spoof')
  if 'live' not in os.listdir(f'{dataPath}/{dir}'):
    os.mkdir(f'{dataPath}/{dir}/live')

os.mkdir(f'{dataPath}/spoof')
os.mkdir(f'{dataPath}/live')

os.mkdir(f'{dataPath}/train')
os.mkdir(f'{dataPath}/train/live')
os.mkdir(f'{dataPath}/train/spoof')

os.mkdir(f'{dataPath}/test')
os.mkdir(f'{dataPath}/test/live')
os.mkdir(f'{dataPath}/test/spoof')

def unify_folders(root_dir):
  spoof_dir = os.path.join(root_dir, 'spoof')
  live_dir = os.path.join(root_dir, 'live')

  for directory in os.listdir(root_dir):
    if not directory.isdigit():
      continue

    spoof_images = os.listdir(os.path.join(root_dir, directory, 'spoof'))
    live_images = os.listdir(os.path.join(root_dir, directory, 'live'))

    for image in spoof_images:
      os.rename(os.path.join(root_dir, directory, 'spoof', image),
                os.path.join(spoof_dir, image))

    for image in live_images:
      os.rename(os.path.join(root_dir, directory, 'live', image),
                os.path.join(live_dir, image))

unify_folders(dataPath)


for dir in os.listdir(dataPath):
    if dir.isdigit():
        shutil.rmtree(f'{dataPath}/{dir}')

# rename every file inside spoof and live,
# to contain its nature inside it

for img in get_images(f'{dataPath}/spoof'):
  if random.uniform(0.0, 1.0) > 0.2:
      new_path = os.path.join('< some path >/test/spoof', os.path.basename(img))
  else:
      new_path = os.path.join('< some path >/train/spoof', os.path.basename(img))
  os.rename(img, new_path)

for img in get_images(f'{dataPath}/live'):
  if random.uniform(0.0, 1.0) > 0.2:
      new_path = os.path.join('< some path >/test/live', os.path.basename(img))
  else:
      new_path = os.path.join('< some path >/train/live', os.path.basename(img))
  os.rename(img, new_path)

shutil.rmtree(f'{dataPath}/spoof')
shutil.rmtree(f'{dataPath}/live')