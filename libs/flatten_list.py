def flatten_list(lst):
    flattened = []
    for item in lst:
        if isinstance(item, str):
            flattened.append(item)
        elif isinstance(item, list):
            flattened.extend(flatten_list(item))
    return flattened
