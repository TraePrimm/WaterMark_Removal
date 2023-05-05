import os
import shutil

# Create directories
train_dir = 'root/train'
test_dir = 'root/test'
val_dir = 'root/val'
os.makedirs(train_dir+'/watermark', exist_ok=True)
os.makedirs(train_dir+'/non_watermark', exist_ok=True)
os.makedirs(test_dir+'/watermark', exist_ok=True)
os.makedirs(test_dir+'/non_watermark', exist_ok=True)
os.makedirs(val_dir+'/watermark', exist_ok=True)
os.makedirs(val_dir+'/non_watermark', exist_ok=True)

# Copy images to appropriate directories
src_dir = 'cropped'
files = os.listdir(src_dir)
num_files = len(files)
for i, file in enumerate(files):
    if i < num_files * 0.6:
        if 'nwm' in file:
            shutil.copy(src_dir+'/'+file, train_dir+'/non_watermark')
        else:
            shutil.copy(src_dir+'/'+file, train_dir+'/watermark')
    elif num_files *.6 < i < num_files * 0.8 :
        if 'nwm' in file:
            shutil.copy(src_dir+'/'+file, val_dir+'/non_watermark')
        else:
            shutil.copy(src_dir+'/'+file, val_dir+'/watermark')
    else:
        if 'nwm' in file:
            shutil.copy(src_dir+'/'+file, test_dir+'/non_watermark')
        else:
            shutil.copy(src_dir+'/'+file, test_dir+'/watermark')