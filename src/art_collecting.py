"""
This script scrapes the Fat Cap website for street art images and saved the images and their meta data.
"""

from bs4 import BeautifulSoup
import requests
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 1000)
import boto3
import os 
from datetime import datetime
startTime = datetime.now()
from collections import Counter


def get_all_links(page_start, page_end=760):
    """
    Goes through each page in the 'All Gallery' and collects the links to every individual art page. Returns a list of links.
    """
    page_links = []
    #Loops through every page in the all section to get all the individual artwork links
    for i in range(page_start,page_end+1):
        print(f"Collecting Page {i}")
        base_website = "https://www.fatcap.com/graffiti-pictures.html?page=" + str(i)
        
        page = requests.get(base_website)
        soup = BeautifulSoup(page.content, 'html.parser')

        #Gets all the links to individual graffiti pages
        links = [pic['href'] for pic in soup.select('a[href]') if pic['href'].startswith('/graffiti/') ]

        page_links.extend(links)
    return page_links

#Goes to every individual page to get meta data and images

def scrape_pages(page_links, directory_location=""):
    df = pd.DataFrame(columns=['Title', 'Date', 'Added', 'Poster', 'Artist', 'City', 'Type', 'Support', 'Style', 'Picture_Link'])
    for i in page_links:   
        single_picture = "https://www.fatcap.com" + i 
        
        print(f"Scraping : {single_picture}") 
        try:
            page = requests.get(single_picture)
            soup = BeautifulSoup(page.content, 'html.parser')
            html = list(soup.children)[2]
            body = list(html.children)[3]

            #Dictionary to append to Pandas
            attribute_dict = dict()

            #Gets section of page with meta-data
            picture_attributes = body.select('div[class=description] > div')

            
            # -1 is to get rid of the footer found on each page
            for att in range(len(picture_attributes)-1):
                #Getting the string, cleaning it, and saving results
                string = picture_attributes[att].get_text()
                split_string = string.replace("\n", "").split(":")
                att_name = split_string[0].strip(" ")
                att_val = split_string[1].strip(" ")
                attribute_dict[att_name] = att_val

            

            #Picture link
            picture = body.find(href=True, id='mainpic')
            
            # if attribute_dict['Type'] == 'Big Walls':
            picture_link = "https://www.fatcap.com" + picture['href'] 
            # else:
            #     picture_link = picture['href']
            
            attribute_dict['Picture_Link'] = picture_link

            df = df.append(attribute_dict, ignore_index=True)
        except:
            continue
    return df

def clean_url(url):
    """
    I did not realize there were multiple domains when I scraped this website. This function corrects the url.
    """

    if 'www.fatcap.org' in url or "s3.amazonaws" in url or "imgfc.com" in url:
        url = url.replace('https://www.fatcap.com',"", 1)
   
    return url
   
def save_images(image_url_and_title, style, df):
    

    folder_name = style.lower().replace(" ", "_")
    
    for i in range(len(image_url_and_title)):

        # print(f"URL?: {image_url_and_title['Picture_Link'][i]}")
        url = clean_url(image_url_and_title['Picture_Link'][i])
        
        title = image_url_and_title['Title'][i].lower().replace(" ", "_").replace(",", "").replace("/", "_")
        # title += str(image_url_and_title['index'][i])
        response = requests.get(url, stream=True)

        print(f"Collecting {i} - Style {style}: {title}")
        print(f"URL : {url}")

        if response.status_code == 200:
            file_string = "data/backup_images/" + folder_name + "/" + title + str(image_url_and_title['index'][i]) + ".jpg"
            print(f"{file_string}")

            with open(file_string, 'wb') as f:
                f.write(response.content)
            
            #Updating dataframe with cleaned URL and File Path        
            df.loc[image_url_and_title['index'][i], 'File_Path'] = file_string
            df.loc[image_url_and_title['index'][i], 'Picture_Link'] = url
            
    return df

if __name__ == '__main__':
    #Collects urls and saves them to csv
    page_links = get_all_links(1, 760)
    df_links = pd.DataFrame(page_links)
    df_links.to_csv("/Users/mt/Galvanize/capstones/street_art_classifier/data/link_list.csv")
    
    #Loops through urls and saves the images urls and meta data
    df = scrape_pages(page_links)
    df.to_csv("/Users/mt/Galvanize/capstones/street_art_classifier/data/meta_data.csv")

    #Scrapes actual images
    df_meta = pd.read_csv("data/meta_data_cleaned.csv", index_col=0)
    
    style = 'Abstract'

    #Gets the subset of images that are done on Walls and the corresponding style. Broke this into two queries to avoid a warning.
    walls_subset = df_meta[df_meta['Support'] == 'Walls']
    image_url_and_title = walls_subset[walls_subset['Style'] == style][['Title', 'Picture_Link']].reset_index()

    df_cleaned = save_images(image_url_and_title, style, df_meta)
    df_cleaned.to_csv("/Users/mt/Galvanize/capstones/street_art_classifier/data/meta_data_cleaned.csv")

    print(datetime.now() - startTime)
