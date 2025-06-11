import os

# serve per rinominare i file che contengono '559' in '560'


def rename_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if '559' in filename:
                old_path = os.path.join(dirpath, filename)
                new_filename = filename.replace('559', '560')
                new_path = os.path.join(dirpath, new_filename)
                
                os.rename(old_path, new_path)
                print(f"Rinominato: {old_path} -> {new_path}")
                
root_dir = "d:/Utente/Desktop/Mados proj/data/MADOS"
rename_files(root_dir)