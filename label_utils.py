def shift_list(lst, item):
    """Shifts a list in the index of the item selected"""
    idx = lst.index(item)
    return lst[idx:] + lst[:idx]
