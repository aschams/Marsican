
def check_train_test_sizes(train_size, test_size):
    """
    Checks to make sure train and test sizes passed are valid.
    Train and test sizes must be floats between 0 and 1 (exclusive) or None.
    If both values are None, an error is raised.
    If train and test size sum to greater than 1, an error is raised.
    If train or test size is None, its value is calculated by subtracting the other value from 1.

    Inputs:
        train_size: float between 0 and 1 (exclusive) or None
        test_size: float between 0 and 1 (exclusive) or None
    Returns:
        train_size: A valid training set size
        test_size: A valid test set size
    """

    if isinstance(train_size, int):
        raise ValueError ("Train size must be a float between 0 and 1(exclusive)")
    if isinstance(test_size, int):
        raise ValueError ("Test size must be a float between 0 and 1(exclusive)")
    if isinstance(test_size, float) and (test_size >=1 or train_size <= 0):
        raise ValueError ("Test size must be between 0 and 1(exclusive)")
    if isinstance(train_size, float) and (train_size >=1 or train_size <= 0):
        raise ValueError ("Train size must be between 0 and 1(exclusive)")
    if (train_size == None) and (test_size == None):
        raise ValueError ("Train or test size must be specified.")
    if (train_size == None) and (test_size != None):
        train_size = 1 - test_size
    if (train_size != None) and (test_size == None):
        test_size = 1 - train_size
    if train_size + test_size > 1:
        raise ValueError ("Train and test size must sum to less than 1.")
    return train_size, test_size

def deep_find_dict(dict_: dict, key, default_val=-1) -> list:
    """
    Finds and returns the associated value of the key in a nested dictionary.
    If a key appears multiple times in a nested dictionary, it returns the first value.
    If the key does not appear in the nested dictionary, returns default_val
    Inputs:
        dict_: Target dictionary
        key: Target key name
        default_val(default = -1): value returned if key does not appear in the nested dictionary
    Returns:
        res: value of first appearance of associated key in the given dictionary
    """
    if key in dict_.keys():
        return dict_[key]
    else:
        for int_key, val in dict_.items():
            if isinstance(dict_[int_key], dict):
                res = deep_find_dict(dict_[int_key], key)
                if res is not None: return res
    return default_val

def extract_paths_with_colonies(json_: dict, min_colonies: int=1, max_colonies: int=-1, path: str='Relative Path'):
    """
    Returns relative file paths of all files with non-zero colony counts
    Inputs:
        json_: JSON file containing the metadata of the plate photographs.
        min_colonies: minimum number of colonies desired on returned images.
        max_colonies: maximum number of colonies desired on returned images.
    Returns:
        img_paths: list of tuples of the colony count and the relative file paths of images containing between
            min_colonies and max_colonies number of colonies.
    """
    img_paths = list()
    if max_colonies == -1:
        max_colonies = np.Inf
    for key, val in json_.items():
        colony_count = deep_find_dict(val, 'Bacterial Load Estimation')
        if (colony_count >= min_colonies) and (colony_count <= max_colonies):
            img_paths.append((colony_count, val['Relative Path']))
    return img_paths

def flatten(test_list):
    """
    https://stackoverflow.com/questions/12472338/flattening-a-list-recursively
    Recursively flattens a nested listed into a list of non-list elements.
    Inputs:
        test_list: list to be flattened
    Returns:
        A flattened version of test_list, containing all the elements contained in test_list and its list elements
        flattened.
    """
    if isinstance(test_list, list):
        if len(test_list) == 0:
            return []
        first, rest = test_list[0], test_list[1:]
        return flatten(first) + flatten(rest)
    else:
        return [test_list]

def make_folders(directory:str, folder_names:list):
    """
    Makes folders in the specified directory.
    Inputs:
        directory: directory to make the folders in
        folder_names: names of the folders to make
    """
    if 'win32' in sys.platform:
        directory = directory + '\\'
    for folder_name in folder_names:
        os.mkdir(directory +  folder_name)

def train_validation_imgsplit(wd,
                              train_size=None,
                              val_size=None,
                              training_dir:str='training',
                              val_dir:str='validation',
                              file_ext:str='.png'):
    """
    Splits data from folders inside a directory into training and validation sets, and moves them
    into training and validation folders.
    It does so by finding all files of specified extension in 1 nested layer of folders in the directory.
    Inputs:
        wd: working directory containing the folders
        train_size: proportion of the data to put in the training set
        test_size: proportion of the data to put in the validation set
        training_dir: directory to place the training data
        val_dir: directory to place the test data
        file_ext: file extension of the files to split into train/validation and move.
    """
    train_sz, test_sz = check_train_test_sizes(train_size, val_size)

    if 'win32' in sys.platform:
        all_images = glob.glob("*\\*" + file_ext)
        wd = wd + '\\'
        training_dir = training_dir + '\\'
        val_dir = val_dir + '\\'

    training_size = int(len(all_images) * train_sz)
    val_size = int(len(all_images) * test_sz)

    train_val_files = np.random.choice(all_images, size=training_size + val_size, replace=False)
    train_files = train_val_files[:training_size]
    val_files = train_val_files[training_size:]

    for train_file in train_files:
        shutil.copyfile(wd + train_file, wd + training_dir + train_file)

    for val_file in val_files:
        shutil.copyfile(wd + val_file, wd + val_dir + val_file)
