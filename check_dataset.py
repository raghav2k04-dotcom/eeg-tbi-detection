import os

dataset_path = r"C:\Users\yogav\OneDrive\Desktop\EEG_PROJECT"

count = 0

for root, dirs, files in os.walk(dataset_path):
    for file in files:

        if file.endswith(".set"):
            print("Found EEG file:", os.path.join(root, file))
            count += 1

print("Total .set files found:", count)
