import requests
import json
import os

url_fromList = 'https://test.jacq.org/development/jacq-services/rest/objects/specimens/fromList'
url_fromFile = 'https://test.jacq.org/development/jacq-services/rest/objects/specimens/fromFile'

# GET THE NAMES OF THE IMAGES 

folder_path_general = "../jp2/"

folderPath_1 = folder_path_general + "141114"
folderPath_2 = folder_path_general + "150120"
folderPath_3 = folder_path_general + "150316"
folderPath_4 = folder_path_general + "150423"
folderPath_5 = folder_path_general + "161103"


imgs_1 = os.listdir(folderPath_1)
imgs_2 = os.listdir(folderPath_2)
imgs_3 = os.listdir(folderPath_3)
imgs_4 = os.listdir(folderPath_4)
imgs_5 = os.listdir(folderPath_5)


# CONNECT ALL 5 IMAGE LISTS INTO 1 (5 FOLDERS)
final_imgs = imgs_1 + imgs_2 + imgs_3 + imgs_4 + imgs_5

params = final_imgs

# SEND POST REQUEST TO URL 
resp = requests.post(url=url_fromList, json=params)


data = resp.json() # Check the JSON Response Content documentation below

# WRITE THE DATA TO A FOLDER
with open('../GroundTruth/ground_truth_json.json', 'w') as f:
    json.dump(data, f)


