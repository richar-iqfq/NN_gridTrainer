def get_id_from_filename(filename: str) -> str:
    '''
    Get training ID from script filename
    '''
    replacements = ['main_', '.py']
    filename = filename.rsplit('/')[-1]

    ID = filename
    for replacement in replacements:
        ID = ID.replace(replacement, '')

    return ID