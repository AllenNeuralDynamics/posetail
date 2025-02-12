import argparse 
import glob
import os 
import shutil
import wget 
import zipfile 


def parse_args(): 

    parser = argparse.ArgumentParser()

    parser.add_argument('--dest', default = './', 
        help = 'path to download rat7m dataset')

    args = parser.parse_args()

    return args


def safe_make(path): 

    if not os.path.exists(path):
        os.makedirs(path)

    return path


def download_zip(url, dest):        
        
    wget.download(url, out = dest)
    zip_name = os.path.basename(os.path.split(os.path.split(url)[0])[0])
    zip_path = os.path.join(dest, f'{zip_name}.zip')

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest)

    os.remove(zip_path)


def download_data(urls, dest = './'): 

    dest = safe_make(dest)

    for url in urls: 
        download_zip(url, dest)


def download_data(urls, dest = './'): 

    dest = safe_make(dest)

    for url in urls: 

        download_zip(url, dest)

        video_paths = glob.glob(os.path.join(dest, '*.mp4'))

        if len(video_paths) > 0: 

            session_name = ('-').join(video_paths[0].split('-')[:2])
            session_dest = safe_make(os.path.join(dest, session_name))

            for video in video_paths:
                shutil.move(video, session_dest)

if __name__ == '__main__': 

    args = parse_args()

    data_urls = [
        'https://figshare.com/ndownloader/articles/13739233/versions/2'
    ]

    video_urls = [
        'https://figshare.com/ndownloader/articles/13764208/versions/1',
        'https://figshare.com/ndownloader/articles/13767502/versions/2',
        'https://figshare.com/ndownloader/articles/13769833/versions/1',
        'https://figshare.com/ndownloader/articles/13770565/versions/1',
        'https://figshare.com/ndownloader/articles/13759336/versions/1',
        'https://figshare.com/ndownloader/articles/13753417/versions/1',
        'https://figshare.com/ndownloader/articles/13751023/versions/1'
    ]

    download_data(urls = data_urls, dest = f'{args.dest}/data')
    download_data(urls = video_urls, dest = f'{args.dest}/videos')
