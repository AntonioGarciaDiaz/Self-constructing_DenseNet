import sys
import os
import urllib.request
import tarfile
import zipfile


def report_download_progress(count, block_size, total_size):
    pct_complete = float(count * block_size) / total_size
    msg = "\r {0:.1%} already downloaded".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()


def extract_data(download_dir):
    print("\nExtracting files")
    for filename in os.listdir(download_dir):
        file_path = os.path.join(download_dir, filename)
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(download_dir)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)


def download_data_url(url, download_dir):
    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir, filename)

    if not os.path.exists(file_path):
        os.makedirs(download_dir, exist_ok=True)
        print("Download %s to %s" % (url, file_path))
        file_path, _ = urllib.request.urlretrieve(
            url=url,
            filename=file_path,
            reporthook=report_download_progress)
        extract_data(download_dir)
