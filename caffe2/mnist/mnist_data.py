import os
import shutil

# This section preps your image and test set in a lmdb database
def DownloadResource(url, path):
    '''Downloads resources from s3 by url and unzips them to the provided path'''
    #import requests, zipfile, StringIO
    import requests, zipfile
    from io import BytesIO
    print("Downloading... {} to {}".format(url, path))
    r = requests.get(url, stream=True)
    z = zipfile.ZipFile(BytesIO(r.content))
    z.extractall(path)
    print("Completed download and extraction.")

def DownloadMNIST():
    current_folder = './'
    data_folder = os.path.join(current_folder, 'tutorial_data', 'mnist')
    root_folder = os.path.join(current_folder, 'tutorial_files', 'tutorial_mnist')
    db_missing = False
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)   
        print("Your data folder was not found!! This was generated: {}".format(data_folder))
    
    # Look for existing database: lmdb
    if os.path.exists(os.path.join(data_folder,"mnist-train-nchw-lmdb")):
        print("lmdb train db found!")
    else:
        db_missing = True
        
    if os.path.exists(os.path.join(data_folder,"mnist-test-nchw-lmdb")):
        print("lmdb test db found!")
    else:
        db_missing = True
    
    # attempt the download of the db if either was missing
    if db_missing:
        print("one or both of the MNIST lmbd dbs not found!!")
        db_url = "http://download.caffe2.ai/databases/mnist-lmdb.zip"
        try:
            DownloadResource(db_url, data_folder)
        except Exception as ex:
            print("Failed to download dataset. Please download it manually from {}".format(db_url))
            print("Unzip it and place the two database folders here: {}".format(data_folder))
            raise ex
    
    if os.path.exists(root_folder):
        print("Looks like you ran this before, so we need to cleanup those old files...")
        shutil.rmtree(root_folder)
        
    os.makedirs(root_folder)

    return root_folder, data_folder
