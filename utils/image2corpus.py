from pytesseract import pytesseract
import os
import glob

# TODO: Uncomment the section to implement it.

# !*****************************************************************************************! #

# PDF to HOCR

# os.chdir(r'C:\Users\lucki\WORK\OFFICE\VAST\Projects\Invoice\Document-Classificaton\NLP\data\layoutlm-format\images\others')
# write_dir = r'C:\Users\lucki\WORK\OFFICE\VAST\Projects\Invoice\Document-Classificaton\NLP\data'

# cat = glob.glob('*.jpg')

# index = 1
# for i in cat:
#     print(i)
#     hocr = pytesseract.image_to_pdf_or_hocr(i, extension='hocr')
#     with open(str(index) + '_.html', 'w+b') as f:
#         f.write(hocr)
#     index += 1


# !*****************************************************************************************! #

# Create Labels.txt
#

os.chdir(r'C:\Users\lucki\WORK\OFFICE\VAST\Projects\Invoice\Document-Classificaton\NLP\data\layoutlm-format\images')
invoice = glob.glob('invoice\*.html')
others = glob.glob('others\*.html')
bill = glob.glob('bill\*.html')
remittance = glob.glob('remittance\*.html')
purchase_order = glob.glob('Purchase Order\*.html')

rows = []
for i in invoice:
    y = i + ' 0'
    print(y)
    rows.append(y)

for i in bill:
    y = i + ' 1'
    print(y)
    rows.append(y)

for i in remittance:
    y = i + ' 2'
    print(y)
    rows.append(y)

for i in purchase_order:
    y = i + ' 3'
    print(y)
    rows.append(y)

for i in others:
    y = i + ' 4'
    print(y)
    rows.append(y)

import random

print(rows)
random.shuffle(rows)

f = open('train.txt', 'w')
for ele in rows[:(len(rows) - 20)]:
    f.write(ele + '\n')
f.close()

h = open('val.txt', 'w')
for ele in rows[(len(rows) - 20):]:
    h.write(ele + '\n')
h.close()
