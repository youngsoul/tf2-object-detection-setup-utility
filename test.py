import os
import pathlib

if __name__ == '__main__':

    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    print(this_file_dir)
    if pathlib.Path(f"{this_file_dir}/tf_util_fixed.txt").exists():
        print("tf_util.py already updated....")
