import subprocess
import pathlib

input_folder = "/Users/jaklin/abtab/data/transcriber/diplomatic/in"
failed_files = []

"""
Wrapper script to run abtab transcriber on all files in a folder.

"""


for file_path in pathlib.Path(input_folder).glob("*"):
    if not str(file_path).endswith(".tbp"):
        continue
    result = subprocess.run(
        [
            "abtab",
            "transcriber",
            "-l",
            "d",  # "diplomatic"
            "-s",
            "s",  # one system
            "-t",
            "n",  # no tablature
            # "-k",
            # "0",  # key signature without accidentals
            "-f",
            str(file_path),
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"**FAILED**: {file_path}")
        failed_files.append(str(file_path))
    else:
        print(f"SUCCESS: {file_path}")

if failed_files:
    print("\nThe following files failed:")
    for f in failed_files:
        print(f)
else:
    print("\nAll files processed successfully.")
