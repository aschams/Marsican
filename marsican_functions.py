
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
