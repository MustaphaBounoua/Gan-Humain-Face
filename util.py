import numpy as np
import os
import cv2
import pandas as pd
import tarfile


ATTRS_NAME = "./data/lfw_attributes.txt"  # http://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt
IMAGES_NAME = "./data/lfw-deepfunneled.tgz"  # http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz
RAW_IMAGES_NAME = "./data/lfw.tgz"  # http://vis-www.cs.umass.edu/lfw/lfw.tgz




def download_files():
    import os 

    FOLDER_DATA_RAW = "./data/" 
    FOLDER_MODEL="./model/"

    if not os.path.exists(FOLDER_DATA_RAW):
        os.mkdir(FOLDER_DATA_RAW)
    if not os.path.exists(FOLDER_MODEL):
        os.mkdir(FOLDER_MODEL)
        
        
    ATTRS_NAME_URL = "http://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt"
    IMAGES_NAME_URL = "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz"
    RAW_IMAGES_NAME_URL ="http://vis-www.cs.umass.edu/lfw/lfw.tgz"

    if not os.path.exists(ATTRS_NAME):
        download_file(ATTRS_NAME_URL,ATTRS_NAME) 
    if not os.path.exists(IMAGES_NAME):
        download_file(IMAGES_NAME_URL,IMAGES_NAME) 
    if not os.path.exists(RAW_IMAGES_NAME):
        download_file(RAW_IMAGES_NAME_URL,RAW_IMAGES_NAME) 






def decode_image_from_raw_bytes(raw_bytes):
    img = cv2.imdecode(np.asarray(bytearray(raw_bytes), dtype=np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_lfw_dataset(
        use_raw=False,
        dx=80, dy=80,
        dimx=45, dimy=45):

    # read attrs
    df_attrs = pd.read_csv(ATTRS_NAME, sep='\t', skiprows=1)
    df_attrs.columns = list(df_attrs.columns)[1:] + ["NaN"]
    df_attrs = df_attrs.drop("NaN", axis=1)
    imgs_with_attrs = set(map(tuple, df_attrs[["person", "imagenum"]].values))

    # read photos
    all_photos = []
    photo_ids = []

    with tarfile.open(RAW_IMAGES_NAME if use_raw else IMAGES_NAME) as f:
        for m in f.getmembers():
            if m.isfile() and m.name.endswith(".jpg"):
                # prepare image
                img = decode_image_from_raw_bytes(f.extractfile(m).read())
                img = img[dy:-dy, dx:-dx]
                img = cv2.resize(img, (dimx, dimy))
                # parse person
                fname = os.path.split(m.name)[-1]
                fname_splitted = fname[:-4].replace('_', ' ').split()
                person_id = ' '.join(fname_splitted[:-1])
                photo_number = int(fname_splitted[-1])
                if (person_id, photo_number) in imgs_with_attrs:
                    all_photos.append(img)
                    photo_ids.append({'person': person_id, 'imagenum': photo_number})

    photo_ids = pd.DataFrame(photo_ids)
    all_photos = np.stack(all_photos).astype('uint8')

    # preserve photo_ids order!
    all_attrs = photo_ids.merge(df_attrs, on=('person', 'imagenum')).drop(["person", "imagenum"], axis=1)

    return all_photos, all_attrs





# https://www.saltycrane.com/blog/2009/11/trying-out-retry-decorator-python/
def retry(ExceptionToCheck, tries=4, delay=3, backoff=2):
    def deco_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except KeyboardInterrupt as e:
                    raise e
                except ExceptionToCheck as e:
                    print("%s, retrying in %d seconds..." % (str(e), mdelay))
                    traceback.print_exc()
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)
        return f_retry  # true decorator
    return deco_retry



@retry(Exception)
def download_file(url, file_path):
    """Download file

    Args:
        url ([str]): [Url of the file]
        file_path ([File_path]): [Where to save the file]

    Raises:
        Exception: [KeyboardInterrupt or  ExceptionToCheck ]
    """
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length'))
    
    incomplete_download = False
    try:
        with open(file_path, 'wb', buffering=16 * 1024 * 1024) as f:
            for chunk in tqdm(r.iter_content(4 * 1024 * 1024) ):
                f.write(chunk)
    except Exception as e:
        raise e
    finally:
        if os.path.exists(file_path) and os.path.getsize(file_path) != total_size:
            incomplete_download = True
            os.remove(file_path)
    if incomplete_download:
        raise Exception("Incomplete download")