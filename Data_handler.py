'''
    A module that holds the DataLoader class used for loading datasets into the training models.
'''

## --- Imports --- ##
import requests
import numpy as np
import io
import sys

class DataLoader:
    '''
    A DataLoader class for loading datasets and making them \n
    available to the ML model.
    '''
    def __init__(self):
        pass
        # Maybe something will go here

    @property
    def train(self):
        return self._train
    
    @property
    def test(self):
        return self._test
    
    @property
    def labels(self):
        return self._labels

    def _download_from_internet(self, urls:dict):
        '''
        Downloads datasets from given URLs.

        @param urls: dictionary with dataset names as keys and URLs as values.
        @return: tuple of (train_data, test_data, labels)
        '''
        datasets = {}
        for key, url in urls.items():
            if "dropbox.com" in url:
                url = self._guarantee_dropbox_url(url)
            print(f"attempting to download {key} data...")
            response = requests.get(url)
            self._errorhandler(response)  # ensure download worked
            try:
                datasets[key] = np.load(io.BytesIO(response.content)) # Dump data into memory then load to numpy array
            except ValueError:
                # If .npy loading fails, try loading as pickle (for .pkl files)
                print(f"Warning: {url} is pickled, and may be harmful", file=sys.stderr)
                datasets[key] = np.load(io.BytesIO(response.content), allow_pickle=True)
        return datasets["train"], datasets["test"], datasets["labels"]
    
    def _errorhandler(self, response):
        '''
        A fail safe function to handle errors during data download.
        '''
        try:
            response.raise_for_status()  # This will raise an HTTPError for bad status codes
        except requests.exceptions.HTTPError as err:
            print(f"HTTP Error: {err}")
        except requests.exceptions.ConnectionError as err:
            print(f"Connection Error: {err}")
        except requests.exceptions.Timeout as err:
            print(f"Timeout Error: {err}")
        except requests.exceptions.RequestException as err:
            print(f"An unexpected error occurred: {err}")
        except:
            print("Unexpected error during data download.")
        else:
            print("Downloaded requested data successfuly!")
            return None
        sys.exit()

    def _load_data(self):
        '''
        Abstract method to load dataset
        '''
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def _guarantee_dropbox_url(self, url: str) -> str:
        '''
        Ensure a Dropbox link points to the raw file (dl=1).
        If the link already has query parameters, replace dl=0 with dl=1.
        If no dl parameter exists, append it.
        @param url: any Dropbox URL
        '''
        if "dl=0" in url:
            return url.replace("dl=0", "dl=1")
        elif "dl=1" in url:
            return url  # already correct
        elif "?" in url:
            return url + "&dl=1"
        else:
            return url + "?dl=1"
        

class BlackWhite(DataLoader):
    '''
    DataLoader subclass for loading the black and white dataset.
    '''
    def __init__(self):
        super().__init__()
        self._train, self._test, self._labels = self._load_data()

    def _load_data(self):
        '''
        Loads black and white dataset.
        '''
        urls = {
            "train"  : 'https://www.dropbox.com/scl/fi/fjye8km5530t9981ulrll/mnist_bw.npy?rlkey=ou7nt8t88wx1z38nodjjx6lch&st=5swdpnbr&dl=0',
            "test"   : 'https://www.dropbox.com/scl/fi/dj8vbkfpf5ey523z6ro43/mnist_bw_te.npy?rlkey=5msedqw3dhv0s8za976qlaoir&st=nmu00cvk&dl=0',
            "labels" : 'https://www.dropbox.com/scl/fi/8kmcsy9otcxg8dbi5cqd4/mnist_bw_y_te.npy?rlkey=atou1x07fnna5sgu6vrrgt9j1&st=m05mfkwb&dl=0'
        }
        return super()._download_from_internet(urls)


class Color(DataLoader):   
    '''
    DataLoader subclass for loading the color dataset.
    '''
    def __init__(self):
        super().__init__()
        self._train, self._test, self._labels = self._load_data()

    def _load_data(self):
        '''
        Loads color dataset.
        '''
        urls = {
            "train"  : 'https://www.dropbox.com/scl/fi/w7hjg8ucehnjfv1re5wzm/mnist_color.pkl?rlkey=ya9cpgr2chxt017c4lg52yqs9&st=ev984mfc&dl=0',
            "test"   : 'https://www.dropbox.com/scl/fi/w08xctj7iou6lqvdkdtzh/mnist_color_te.pkl?rlkey=xntuty30shu76kazwhb440abj&st=u0hd2nym&dl=0',
            "labels" : 'https://www.dropbox.com/scl/fi/fkf20sjci5ojhuftc0ro0/mnist_color_y_te.npy?rlkey=fshs83hd5pvo81ag3z209tf6v&st=99z1o18q&dl=0'
        }
        return super()._download_from_internet(urls)
    
class CustomURL(DataLoader): #Maybe useful for testing
    '''
    DataLoader subclass for loading datasets from custom URLs.
    '''
    def __init__(self, **kwargs):
        super().__init__()

        for key in ["train", "test", "labels"]:
            if key not in kwargs:
                raise ValueError(f"Missing required URL for '{key}' dataset.")
            urls = kwargs[key]

        self._train, self._test, self._labels = self._load_data(urls)
    
    def _load_data(self, urls: dict):
        '''
        Loads dataset from custom URLs.
        '''
        return super()._download_from_internet(urls)