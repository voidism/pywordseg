import urllib.request
import sys
import time
import zipfile
import os

def unzip(from_path, to_path):
    with zipfile.ZipFile(from_path, 'r') as zip_ref:
        zip_ref.extractall(to_path)
    os.remove(from_path)

def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration)) if duration != 0 else 0
    percent = int(count * block_size * 100 / total_size) if total_size != 0 else 0
    sys.stdout.write("\rDownload models ...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

def save(url, filename):
    urllib.request.urlretrieve(url, filename, reporthook)

def save_unzip(url, filename):
    urllib.request.urlretrieve(url, filename, reporthook)
    unzip(filename, os.path.abspath(os.path.join(__file__ ,"..")))
    print("\n%s built!"%filename.split('.')[0])

if __name__ == "__main__":
    url = "https://www.dropbox.com/s/eiya6ztmjopprsm/ELMoForManyLangs.zip?dl=1"
    file_name = "ELMoForManyLangs.zip"
    save(url, file_name)
    unzip(file_name, os.path.abspath("."))
    print("ELMoForManyLangs built!")
