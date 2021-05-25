import os
import glob
import tempfile
from PIL import Image
from pdf2image import convert_from_path

os.chdir(r'C:\Users\lucki\WORK\OFFICE\VAST\Projects\Invoice\Document-Classificaton\dataset\s3_dump_(gradian)')
write_dir = r'C:\Users\lucki\WORK\OFFICE\VAST\Projects\Invoice\Document-Classificaton\dataset\s3_converted'

pdfs = glob.glob('*.pdf')


index = 1
for i in pdfs:
    filename = i
    print(filename)
    pil_images = convert_from_path(filename,poppler_path=r'C:\Users\lucki\WORK\OFFICE\VAST\Projects\Invoice\Invoice'
                                                         r'-Data-Extraction\Production\poppler-21.03.0\Library\bin')
    for image in pil_images:
        image.save("p_" + str(index) + ".jpg")
        index += 1


