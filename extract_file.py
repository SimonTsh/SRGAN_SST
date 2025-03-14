import py7zr

# Directory where files will be extracted
data_source = 'di-lab'
data_dir = 'data/%s/' % (data_source)

# Path to the .7z archive
data_filename = "train_1y_Australia2.7z"

# Extract the archive
with py7zr.SevenZipFile(f'{data_dir}{data_filename}', mode='r') as archive:
    archive.extractall(path=data_dir)
