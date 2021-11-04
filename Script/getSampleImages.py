from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import urllib
import requests
from PIL import Image
import io

import os

driver = webdriver.Chrome("chromedriver.exe")


def getGoogleSearchImages(link,folder,file):
    driver.get(link)

    imgs = driver.find_elements_by_tag_name("img")
    print(len(imgs))

    count = 0
    for img in imgs:
        data = img.screenshot_as_png
        imageIo = io.BytesIO(data)
        image = Image.open(imageIo).convert("RGBA")

        savePath = "../Data/Original/"+folder+"/"+file+"_" + "{}.png".format(count)
        count += 1

        image.save(savePath)



urlCat="https://www.google.com/search?q=%E7%8C%AB&tbm=isch&sxsrf=ALeKk03ZFPiIqpPoyzx3SZHN9jYQqIo29w:1628492092231&source=lnms&sa=X&ved=2ahUKEwjU3rTlraPyAhUFBd4KHUwaCNkQ_AUoAnoECAcQBA&biw=1284&bih=1297&dpr=1.5"

getGoogleSearchImages(urlCat,"cats","cat")

urlDog="https://www.google.com/search?q=%E7%8A%AC&tbm=isch&ved=2ahUKEwjX2PW9xqPyAhUJMN4KHb_0B2gQ2-cCegQIABAA&oq=%E7%8A%AC&gs_lcp=CgNpbWcQAzIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQyEQgAEIAEELEDEIMBELEDEIMBMgsIABCABBCxAxCxAzILCAAQgAQQsQMQsQMyCAgAELEDEIMBMgsIABCABBCxAxCxAzILCAAQgAQQsQMQsQM6BwgjEOoCECc6CggAEIAEELEDEAQ6CAgAEIAEELEDOgYIABAEEAM6BwgAEIAEEAQ6DQgAEIAEELEDEIMBEARQ6AVYuRBg6hNoAXAAeACAAWCIAdQCkgEBNJgBAKABAaoBC2d3cy13aXotaW1nsAEDwAEB&sclient=img&ei=IOsQYdfIEYng-Aa_6Z_ABg&bih=1297&biw=1284"

getGoogleSearchImages(urlDog,"dogs","dog")