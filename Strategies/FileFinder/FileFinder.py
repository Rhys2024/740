import os, platform


def get_path(end_item, dir_in_740 = 'Reference_Data'):

    system = platform.system()
    # os.getlogin()
    # Username will be different for different Computers #
    
    if system == "Darwin":
        item_list = [os.path.expanduser('~'), 'Desktop', '740', dir_in_740, end_item]
        path = "/"
    elif system == "Windows":
        
        user = os.environ['USERPROFILE']
        
        
        item_list = [user, 'OneDrive', 'Desktop', '740', dir_in_740, end_item]
        path = "\\"
    for item in item_list:
        path = os.path.join(path, item)
    
    return path