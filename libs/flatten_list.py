def flatten_list(lst):
"""flattens list where each element of teh list can either be a single item (including string) 
    or a sub-list. Strings aren't broken up.
    Arguments:
        lst -- a list
    Returns.
        flattened -- a flat list
    """
    flattened = []
    for item in lst:
        if isinstance(item, str):
            flattened.append(item)
        elif isinstance(item, list):
            flattened.extend(flatten_list(item))
    return flattened
