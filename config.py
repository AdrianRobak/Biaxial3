import os


# Ścieżka dostępu do folderu z plikami MIDI
base = "/Volumes/RobakSSD4/Biaxial/Biaxial/MIDI base/2Debussy-Ravel"
# base = "/Volumes/RobakSSD4/Biaxial/Biaxial/MIDI base/2Debussy-Ravel"
directory_name = os.path.basename(base).split("/")[-1]

# Nazwa zmiennej zawierającej nazwę folderu z plikami MIDI
training_base = directory_name

dropout = 0.5   