import os
import shutil
import glob

os.chdir(r'C:\Users\lucki\WORK\OFFICE\VAST\Invoice\DATASET\rvl-cdip\labels')
with open("train.txt") as file_in:
    lines = []
    for line in file_in:
        lines.append(line)

print(lines[:5])
invoicePath = []

for line in lines:
    if " 11" in line:
        invoicePath.append('images/' + line[:-4])
        print(line[:-4] + ' invoice file path added to list')
        print("******************************************")
print(invoicePath[:5])
print()
print()

if input("Do you want to copy files to folder? Type 'YES' or 'NO' :: ") == 'YES':
    for f in invoicePath:
        shutil.copy(f, 'invoice/')
