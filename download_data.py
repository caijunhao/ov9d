import subprocess
import argparse
import os


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--models', action='store_true') 
parser.add_argument('--multi', action='store_true') 
args = parser.parse_args()

os.makedirs('ov9d', exist_ok=True)
os.chdir('ov9d')
os.makedirs('checkpoints', exist_ok=True)
if not os.path.exists(os.path.join('checkpoints', 'v1-5-pruned-emaonly.ckpt')):
    os.chdir('checkpoints')
    print('Downloading stable-diffusion-v1-5 ...')
    url_sd = 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt'
    subprocess.call(f'wget {url_sd}', shell=True)
    os.chdir('..')

base_url = 'https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/ov9d'

os.makedirs('pretrained_model', exist_ok=True)
if not os.path.exists(os.path.join('pretrained_model', 'model.ckpt')):
    os.chdir('pretrained_model')
    print('Downloading pretrained checkpoint ...')
    subprocess.call(f'wget {base_url}/pretrained_model/model.ckpt', shell=True)
    os.chdir('..')

files = [
    'oo3d9dsingle.txt', 
    'cid2oid.json', 
    'class_list.json', 
    'models_info.json', 
    'models_info_with_symmetry.json', 
    'name2oid.json', 
    'oid2cid.json', 
    'oo3d9dsingle_class_embeddings.pth', 
]

for f in files:
    if os.path.exists(f):
        print(f'File {f} existed, skip download')
        continue
    print(f'Downloading {f} ...')
    url = os.path.join(base_url, f)
    subprocess.call(f'wget {url}', shell=True)
    print(f'File {f} successfully downloaded')

with open('oo3d9dsingle.txt', 'r') as f:
    files = f.readlines()
os.makedirs('oo3d9dsingle', exist_ok=True)
os.chdir('oo3d9dsingle')
for f in files:
    f = f[:-1]
    if os.path.exists(f) or os.path.exists(f.split('.')[0]):
        print(f'File {f} existed, skip download')
        continue
    print(f'Downloading {f} ...')
    url = os.path.join(base_url, 'oo3d9dsingle', f)
    subprocess.call(f'wget {url}', shell=True)
    print(f'File {f} successfully downloaded')
    print(f'Uncompressing {f} ...')
    subprocess.call(f'tar -zxvf {f}', shell=True)
    print(f'File {f} successfully extracted')
    subprocess.call(f'rm {f}', shell=True)
os.chdir('..')

folders = []
if args.multi:
    folders.append('oo3d9dmulti')
if args.models:
    folders.extend(['models', 'models_eval'])

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    os.chdir(folder)
    for i in range(1, 101):
        f = f'batch_{i}.tgz'
        if os.path.exists(f):
            print(f'File {f} existed, skip download')
            continue
        print(f'Downloading {f} ...')
        url = os.path.join(base_url, folder, f)
        subprocess.call(f'wget {url}', shell=True)
        print(f'File {f} successfully downloaded')
        print(f'Uncompressing {f} ...')
        subprocess.call(f'tar -zxvf {f}', shell=True)
        print(f'File {f} successfully extracted')
        subprocess.call(f'rm {f}', shell=True)
        subprocess.call(f"mv batch_{i}/* ./", shell=True)
        subprocess.call(f'rm -r batch_{i}', shell=True)
    os.chdir('..')
