import os
# import shutil
# import glob
#
os.chdir(r'C:\Users\lucki\WORK\OFFICE\VAST\Invoice\s3_images')
#
# # pdfs = glob.glob('*.pdf')
# # print(pdfs)
# # print(len(pdfs))
# # for pdf in pdfs:
# #     shutil.copy(pdf, 'pdfs/')
#
# d = "C:\Users\lucki\WORK\OFFICE\VAST\Invoice\s3"
# for path in os.listdir(d):
#     full_path = os.path.join(d, path)
#     if os.path.isfile(full_path):
#         print(full_path)

from os import path
from glob import glob


def find_ext(dr, ext):
    return glob(path.join(dr, "*.{}".format(ext)))


print(find_ext('.', 'jpg'))
