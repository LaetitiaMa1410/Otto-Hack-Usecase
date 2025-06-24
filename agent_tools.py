from PIL import Image
import cv2
import pandas as pd
import extcolors
from colormap import rgb2hex
import os
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
import time
from dotenv import load_dotenv
from typing import Tuple, List
import pandas as pd


#extract image metadata 
def get_image_size(img_path: str) -> Tuple[int, int]:
    img = cv2.imread(img_path)
    height,width = img.shape[:2]
    return height, width

def get_image_resolution(img_path: str) -> int:
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    return width*height

def get_image_sharpness(img_path: str) -> float:
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()
    return sharpness


#color extraction
def get_image_colors(img_path: str) -> pd.DataFrame:

    height, width = get_image_size(img_path)
    output_width = 900 

    if width > 900:
        img = Image.open(img_path)
        wpercent = (output_width/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((output_width,hsize))
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        img_path = f'C:/Users/laetitiamaar/OneDrive - Microsoft/AI_projects-main/OttoHack/data/{image_name}_resized'
        img.save(img_path)


    colors_x = extcolors.extract_from_path(img_path, tolerance = 12, limit = 12)
    colors_pre_list = str(colors_x).replace('([(','').split(', (')[0:-1]
    df_rgb = [i.split('), ')[0] + ')' for i in colors_pre_list]
    df_percent = [i.split('), ')[1].replace(')','') for i in colors_pre_list]
    
    #convert RGB to HEX code
    df_color_up = [rgb2hex(int(i.split(", ")[0].replace("(","")),
                          int(i.split(", ")[1]),
                          int(i.split(", ")[2].replace(")",""))) for i in df_rgb]
    
    df = pd.DataFrame(zip(df_color_up, df_percent), columns = ['c_code','occurence'])
    return df

def get_image_text(img_path: str) -> List[str]:
    
    # Azure Computer Vision credentials
    load_dotenv()
    endpoint = os.getenv("AZURE_CV_ENDPOINT")
    key = os.getenv("AZURE_CV_KEY")

    # Create client
    client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))

    with open(img_path, "rb") as image_stream:
        read_response = client.read_in_stream(image_stream, raw=True)

    operation_location = read_response.headers["Operation-Location"]
    operation_id = operation_location.split("/")[-1]

    while True:
        result = client.get_read_result(operation_id)
        if result.status not in ['notStarted', 'running']:
            break
        time.sleep(1)

    image_text = []
    if result.status == 'succeeded':
        for page in result.analyze_result.read_results:
            for line in page.lines:
                image_text.append(line.text)

    return image_text


def ocr(img_path: str) -> List[str]:
    
    load_dotenv()
    endpoint = os.getenv("AZURE_DI_ENDPOINT")
    key = os.getenv("AZURE_DI_KEY")

    client = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))

    # Read image and send for analysis
    with open(img_path, "rb") as f:
        poller = client.begin_analyze_document("prebuilt-read", document=f)
        result = poller.result()

    # Extract text lines
    lines = []
    for page in result.pages:
        for line in page.lines:
            lines.append(line.content)

    return lines