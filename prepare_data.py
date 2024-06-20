from pathlib import Path
import os

os.chdir('ov9d')

train_dir = Path('train')
train_dir.mkdir(parents=True, exist_ok=True)
test_dir = Path('test')
test_dir.mkdir(parents=True, exist_ok=True)
single = Path('single')

test_cats = ['bowl', 'bumbag', 'dumpling', 'facial_cream', 'handbag', 
             'litchi', 'mouse', 'pineapple', 'teddy_bear', 'toy_truck']

[(test_dir/cat).mkdir(parents=True, exist_ok=True) for cat in test_cats]
(test_dir/'all').mkdir(parents=True, exist_ok=True)


for folder in os.listdir(single):
    print(folder)
    cat = '_'.join(folder.split('_')[:-2])
    if cat in test_cats:
        os.symlink((single/folder).resolve(), (test_dir/cat/folder).resolve())
        os.symlink((single/folder).resolve(), (test_dir/'all'/folder).resolve())
    else:
        os.symlink((single/folder).resolve(), (train_dir/folder).resolve())
