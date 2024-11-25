# main.py
import os
import base64
import io
import math
from flask import Flask, flash, render_template, Response, redirect, request, session, abort, url_for
import mysql.connector
import hashlib
import datetime
from datetime import datetime
from datetime import date
import random
from random import randint
from urllib.request import urlopen
import webbrowser
import cv2
from math import log10, sqrt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import imagehash
import shutil
from skimage.metrics import structural_similarity
import PIL.Image
from PIL import Image
#from PIL import Image, ImageDraw, ImageFilter
from PIL import Image, ImageFilter, ImageDraw, ImageStat

from werkzeug.utils import secure_filename
import urllib.request
import urllib.parse



mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  charset="utf8",
  use_pure="True",
  database="image_immunizer"

)
app = Flask(__name__)
##session key
app.secret_key = 'abcdef'
#######
UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = { 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#####
@app.route('/', methods=['GET', 'POST'])
def index():
    msg=""
    ff=open("det.txt","w")
    ff.write("1")
    ff.close()

    ff1=open("photo.txt","w")
    ff1.write("1")
    ff1.close()

    #s="welcome"
    #v=s[2:5]
    #print(v)
    
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor(buffered=True)
        cursor.execute('SELECT * FROM im_user WHERE uname = %s AND pass = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('userhome'))
        else:
            msg = 'Incorrect username/password!'
    

    return render_template('index.html',msg=msg)

@app.route('/socialapp1', methods=['GET', 'POST'])
def socialapp1():
    msg=""
    ff=open("det.txt","w")
    ff.write("1")
    ff.close()

    ff1=open("photo.txt","w")
    ff1.write("1")
    ff1.close()

    #s="welcome"
    #v=s[2:5]
    #print(v)  
    
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM im_user WHERE uname = %s AND pass = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('userhome'))
        else:
            msg = 'Incorrect username/password!'
    

    return render_template('index.html',msg=msg)

@app.route('/socialapp2', methods=['GET', 'POST'])
def socialapp2():
    msg=""
    ff=open("det.txt","w")
    ff.write("1")
    ff.close()

    ff1=open("photo.txt","w")
    ff1.write("1")
    ff1.close()

    #s="welcome"
    #v=s[2:5]
    #print(v)
    
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM im_user1 WHERE uname = %s AND pass = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('userhome1'))
        else:
            msg = 'Incorrect username/password!'
    

    return render_template('web/login.html',msg=msg)

@app.route('/login', methods=['GET', 'POST'])
def login():
    msg=""
    ff=open("det.txt","w")
    ff.write("1")
    ff.close()

    ff1=open("photo.txt","w")
    ff1.write("1")
    ff1.close()

    #s="welcome"
    #v=s[2:5]
    #print(v)
    
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM im_user1 WHERE uname = %s AND pass = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('userhome1'))
        else:
            msg = 'Incorrect username/password!'
    

    return render_template('web/login.html',msg=msg)

@app.route('/immunizer', methods=['GET', 'POST'])
def immunizer():
    msg=""
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM im_admin WHERE username = %s AND password = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('admin'))
        else:
            msg = 'Incorrect username/password!'
    

    return render_template('login_admin.html',msg=msg)


@app.route('/login_admin', methods=['GET', 'POST'])
def login_admin():
    msg=""
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM im_admin WHERE username = %s AND password = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('admin'))
        else:
            msg = 'Incorrect username/password!'
    

    return render_template('login_admin.html',msg=msg)


@app.route('/change_pw', methods=['GET', 'POST'])
def change_pw():
    msg=""
    uname=""
    if 'username' in session:
        uname = session['username']

    cursor = mydb.cursor()
    cursor.execute("SELECT * FROM im_user where uname=%s",(uname,))
    data = cursor.fetchone()
    
    if request.method=='POST':
        oldpass=request.form['oldpass']
        newpass=request.form['newpass']
        
        cursor.execute('SELECT count(*) FROM im_user WHERE uname = %s AND pass = %s', (uname, oldpass))
        account = cursor.fetchone()[0]
        if account>0:
            cursor.execute("update im_user set pass=%s where uname=%s",(newpass,uname))
            mydb.commit()
            msg="success"
        else:
            msg = 'fail'
    return render_template('change_pw.html',msg=msg,data=data)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    msg=""
    vid=""
    nam=""
    email=""
    mess=""
    bdata=""
    bc=""
    mycursor = mydb.cursor()
    mycursor.execute("SELECT max(id)+1 FROM im_user")
    maxid = mycursor.fetchone()[0]
    if maxid is None:
        maxid=1
    if request.method=='POST':
        name=request.form['name']
        gender=request.form['gender']
        dob=request.form['dob']
        aadhar=request.form['aadhar']
        mobile=request.form['mobile']
        email=request.form['email']
        uname=request.form['uname']
        pass1=request.form['pass']
        
        mycursor.execute('SELECT count(*) FROM im_user WHERE uname = %s || aadhar=%s', (uname,aadhar))
        cnt = mycursor.fetchone()[0]
        if cnt==0:
            now = datetime.now()
            rdate=now.strftime("%d-%m-%Y")

            
            adr=str(aadhar)
            rn=randint(50,90)
            v1=name[0:2]
            v2=str(rn)
            v3=adr[0:3]
            bkey=v1+str(maxid)+v2+v3

            #f1=open("bc.txt","r")
            #bc=f1.read()
            #f1.close()
            sql = "INSERT INTO im_user(id,name,gender,dob,mobile,email,aadhar,uname,pass,create_date,photo) VALUES (%s,%s, %s, %s, %s, %s,%s,%s,%s,%s,%s)"
            val = (maxid,name,gender,dob,mobile,email,aadhar,uname,pass1,rdate,'')
            mycursor.execute(sql, val)
            mydb.commit()            
            print(mycursor.rowcount, "Registered Success")
            #mess="Dear "+name+", Username: "+uname+", Password: "+pass1+", Block Key: "+bkey
            msg="success"
            #vid=str(maxid)
            #nam="1"
            ###
            #mycursor.execute('SELECT * FROM im_user WHERE uname=%s', (uname,))
            #dd = mycursor.fetchone()
            #dtime=str(dd[11])
            #bdata="ID:"+str(maxid)+", Username:"+uname+", Status:Registered, Aadhar:"+aadhar+", Date: "+dtime
            ###
        else:
            msg='fail'
    return render_template('signup.html',msg=msg)

@app.route('/signup1', methods=['GET', 'POST'])
def signup1():
    msg=""
    vid=""
    nam=""
    email=""
    mess=""
    bdata=""
    bc=""
    mycursor = mydb.cursor()
    mycursor.execute("SELECT max(id)+1 FROM im_user1")
    maxid = mycursor.fetchone()[0]
    if maxid is None:
        maxid=1
    if request.method=='POST':
        name=request.form['name']
        gender=request.form['gender']
        dob=request.form['dob']
        aadhar=request.form['aadhar']
        mobile=request.form['mobile']
        email=request.form['email']
        uname=request.form['uname']
        pass1=request.form['pass']
        
        mycursor.execute('SELECT count(*) FROM im_user1 WHERE uname = %s || aadhar=%s', (uname,aadhar))
        cnt = mycursor.fetchone()[0]
        if cnt==0:
            now = datetime.now()
            rdate=now.strftime("%d-%m-%Y")

            
            adr=str(aadhar)
            rn=randint(50,90)
            v1=name[0:2]
            v2=str(rn)
            v3=adr[0:3]
            bkey=v1+str(maxid)+v2+v3

            #f1=open("bc.txt","r")
            #bc=f1.read()
            #f1.close()
            sql = "INSERT INTO im_user1(id,name,gender,dob,mobile,email,aadhar,uname,pass,create_date,photo) VALUES (%s,%s, %s, %s, %s, %s,%s,%s,%s,%s,%s)"
            val = (maxid,name,gender,dob,mobile,email,aadhar,uname,pass1,rdate,'')
            mycursor.execute(sql, val)
            mydb.commit()            
            print(mycursor.rowcount, "Registered Success")
            #mess="Dear "+name+", Username: "+uname+", Password: "+pass1+", Block Key: "+bkey
            msg="success"
            #vid=str(maxid)
            #nam="1"
            ###
            #mycursor.execute('SELECT * FROM im_user WHERE uname=%s', (uname,))
            #dd = mycursor.fetchone()
            #dtime=str(dd[11])
            #bdata="ID:"+str(maxid)+", Username:"+uname+", Status:Registered, Aadhar:"+aadhar+", Date: "+dtime
            ###
        else:
            msg='fail'
    return render_template('web/register.html',msg=msg)

###Immunizer#####################
def gcr(im, percentage):
    '''basic "Gray Component Replacement" function. Returns a CMYK image with 
       percentage gray component removed from the CMY channels and put in the
       K channel, ie. for percentage=100, (41, 100, 255, 0) >> (0, 59, 214, 41)'''
    cmyk_im = im.convert('CMYK')
    if not percentage:
        return cmyk_im
    cmyk_im = cmyk_im.split()
    cmyk = []
    for i in range(4):
        cmyk.append(cmyk_im[i].load())
    for x in range(im.size[0]):
        for y in range(im.size[1]):
            gray = min(cmyk[0][x,y], cmyk[1][x,y], cmyk[2][x,y]) * percentage / 100
            for i in range(3):
                cmyk[i][x,y] = cmyk[i][x,y] - gray
            cmyk[3][x,y] = gray
    return Image.merge('CMYK', cmyk_im)

def halftone(im, cmyk, sample, scale):
    '''Returns list of half-tone images for cmyk image. sample (pixels), 
       determines the sample box size from the original image. The maximum 
       output dot diameter is given by sample * scale (which is also the number 
       of possible dot sizes). So sample=1 will presevere the original image 
       resolution, but scale must be >1 to allow variation in dot size.'''
    cmyk = cmyk.split()
    dots = []
    angle = 0
    for channel in cmyk:
        channel = channel.rotate(angle, expand=1)
        size = channel.size[0]*scale, channel.size[1]*scale
        half_tone = Image.new('L', size)
        draw = ImageDraw.Draw(half_tone)
        for x in range(0, channel.size[0], sample):
            for y in range(0, channel.size[1], sample):
                box = channel.crop((x, y, x + sample, y + sample))
                stat = ImageStat.Stat(box)
                diameter = (stat.mean[0] / 255)**0.5
                edge = 0.5*(1-diameter)
                x_pos, y_pos = (x+edge)*scale, (y+edge)*scale
                box_edge = sample*diameter*scale
                draw.ellipse((x_pos, y_pos, x_pos + box_edge, y_pos + box_edge), fill=255)
        half_tone = half_tone.rotate(-angle, expand=1)
        width_half, height_half = half_tone.size
        xx=(width_half-im.size[0]*scale) / 2
        yy=(height_half-im.size[1]*scale) / 2
        half_tone = half_tone.crop((xx, yy, xx + im.size[0]*scale, yy + im.size[1]*scale))
        dots.append(half_tone)
        angle += 15
    return dots

def compare(img1,img2):
    # Load images
    before = cv2.imread("static/upload/"+img1)
    after = cv2.imread("static/upload/"+img2)

    # Convert images to grayscale
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between the two images
    (score, diff) = structural_similarity(before_gray, after_gray, full=True)
    print("Image Similarity: {:.4f}%".format(score * 100))
    per=format(score * 100)

    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1] 
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] before we can use it with OpenCV
    diff = (diff * 255).astype("uint8")
    diff_box = cv2.merge([diff, diff, diff])

    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    mask = np.zeros(before.shape, dtype='uint8')
    filled_after = after.copy()
    j=1
    for c in contours:
        area = cv2.contourArea(c)
        if area > 40:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
            mm=cv2.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.imwrite("static/test/ggg.jpg", mm)

            image = cv2.imread("static/test/ggg.jpg")
            cropped = image[y:y+h, x:x+w]
            gg="f"+str(j)+".jpg"
            cv2.imwrite("static/test/"+gg, cropped)
        
            cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.drawContours(mask, [c], 0, (255,255,255), -1)
            cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)
            j+=1
    
    #cv2.imshow('before', before)
    #cv2.imshow('after', after)
    #cv2.imshow('diff', diff)
    #cv2.imshow('diff_box', diff_box)
    #cv2.imshow('mask', mask)
    #cv2.imshow('filled after', filled_after)
    #cv2.waitKey()
    value=[per,j]
    return value

def attack1(img1,img2):
    # Load images
    before = cv2.imread("static/upload/"+img2)
    after = cv2.imread("static/upload/"+img1)

    # Convert images to grayscale
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between the two images
    (score, diff) = structural_similarity(before_gray, after_gray, full=True)
    print("Image Similarity: {:.4f}%".format(score * 100))
    per=format(score * 100)

    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1] 
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] before we can use it with OpenCV
    diff = (diff * 255).astype("uint8")
    diff_box = cv2.merge([diff, diff, diff])

    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    mask = np.zeros(before.shape, dtype='uint8')
    filled_after = after.copy()
    j=1
    for c in contours:
        area = cv2.contourArea(c)
        if area > 40:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
            mm=cv2.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.imwrite("static/test/ggg.jpg", mm)

            image = cv2.imread("static/test/ggg.jpg")
            cropped = image[y:y+h, x:x+w]
            gg="h"+str(j)+".jpg"
            cv2.imwrite("static/test/"+gg, cropped)
        
            cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.drawContours(mask, [c], 0, (255,255,255), -1)
            cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)
            j+=1

    #########
    e=j-1
    n=1
    y=0
    k=0
    while n<=e:
        main_image = cv2.imread('static/upload/'+img1)
        gray_image = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
        #open the template as gray scale image
        ffn="h"+str(n)+".jpg"
        template = cv2.imread("static/test/"+ffn, 0)
        width, height = template.shape[::-1] #get the width and height
        #match the template using cv2.matchTemplate
        match = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        position = np.where(match >= threshold) #get the location of template in the image
        k=0
        for point in zip(*position[::-1]): #draw the rectangle around the matched template
           cv2.rectangle(main_image, point, (point[0] + width, point[1] + height), (0, 204, 153), 0)
           k+=1
        if k>1:
            y+=k
            
        n+=1
    ########
    #cv2.imshow('before', before)
    #cv2.imshow('after', after)
    #cv2.imshow('diff', diff)
    #cv2.imshow('diff_box', diff_box)
    #cv2.imshow('mask', mask)
    #cv2.imshow('filled after', filled_after)
    #cv2.waitKey()
    value=[per,j,y]
    return value

def attack11(img1,img2):
    # Load images
    before = cv2.imread("static/upload/"+img2)
    after = cv2.imread("static/upload/"+img1)

    # Convert images to grayscale
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between the two images
    (score, diff) = structural_similarity(before_gray, after_gray, full=True)
    print("Image Similarity: {:.4f}%".format(score * 100))
    per=format(score * 100)

    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1] 
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] before we can use it with OpenCV
    diff = (diff * 255).astype("uint8")
    diff_box = cv2.merge([diff, diff, diff])

    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    mask = np.zeros(before.shape, dtype='uint8')
    filled_after = after.copy()
    j=1
    for c in contours:
        area = cv2.contourArea(c)
        if area > 40:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
            mm=cv2.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.imwrite("static/test/ggg.jpg", mm)

            image = cv2.imread("static/test/ggg.jpg")
            cropped = image[y:y+h, x:x+w]
            gg="u"+str(j)+".jpg"
            cv2.imwrite("static/test/"+gg, cropped)
        
            cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.drawContours(mask, [c], 0, (255,255,255), -1)
            cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)
            j+=1

    #########
    e=j-1
    n=1
    y=0
    k=0
    while n<=e:
        main_image = cv2.imread('static/upload/'+img2)
        gray_image = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
        #open the template as gray scale image
        ffn="u"+str(n)+".jpg"
        template = cv2.imread("static/test/"+ffn, 0)
        width, height = template.shape[::-1] #get the width and height
        #match the template using cv2.matchTemplate
        match = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        position = np.where(match >= threshold) #get the location of template in the image
        k=0
        for point in zip(*position[::-1]): #draw the rectangle around the matched template
           cv2.rectangle(main_image, point, (point[0] + width, point[1] + height), (0, 204, 153), 0)
           k+=1
        if k>1:
            y+=k
            
        n+=1
    ########
    #cv2.imshow('before', before)
    #cv2.imshow('after', after)
    #cv2.imshow('diff', diff)
    #cv2.imshow('diff_box', diff_box)
    #cv2.imshow('mask', mask)
    #cv2.imshow('filled after', filled_after)
    #cv2.waitKey()
    cv2.imwrite("static/test/p1.jpg", before)
    cv2.imwrite("static/test/p2.jpg", after)
    cv2.imwrite("static/test/p3.jpg", diff)
    cv2.imwrite("static/test/p4.jpg", diff_box)
    cv2.imwrite("static/test/p5.jpg", mask)
    cv2.imwrite("static/test/p6.jpg", filled_after)
        
    value=[per,j,y]
    return value

def attack2(img1,img2):
    # Load images
    before = cv2.imread("static/upload/"+img1)
    after = cv2.imread("static/upload/"+img2)

    # Convert images to grayscale
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between the two images
    (score, diff) = structural_similarity(before_gray, after_gray, full=True)
    print("Image Similarity: {:.4f}%".format(score * 100))
    per=format(score * 100)

    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1] 
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] before we can use it with OpenCV
    diff = (diff * 255).astype("uint8")
    diff_box = cv2.merge([diff, diff, diff])

    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    mask = np.zeros(before.shape, dtype='uint8')
    filled_after = after.copy()
    j=1
    for c in contours:
        area = cv2.contourArea(c)
        if area > 40:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
            mm=cv2.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.imwrite("static/test/ggg.jpg", mm)

            image = cv2.imread("static/test/ggg.jpg")
            cropped = image[y:y+h, x:x+w]
            gg="f"+str(j)+".jpg"
            cv2.imwrite("static/test/"+gg, cropped)
        
            cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.drawContours(mask, [c], 0, (255,255,255), -1)
            cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)
            j+=1

    #########
    e=j-1
    n=1
    y=0
    k=0
    while n<=e:
        main_image = cv2.imread('static/upload/'+img1)
        gray_image = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
        #open the template as gray scale image
        ffn="f"+str(n)+".jpg"
        template = cv2.imread("static/test/"+ffn, 0)
        width, height = template.shape[::-1] #get the width and height
        #match the template using cv2.matchTemplate
        match = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        position = np.where(match >= threshold) #get the location of template in the image
        k=0
        for point in zip(*position[::-1]): #draw the rectangle around the matched template
           cv2.rectangle(main_image, point, (point[0] + width, point[1] + height), (0, 204, 153), 0)
           k+=1
        if k>1:
            y+=k
            
        n+=1
    ########
    #cv2.imshow('before', before)
    #cv2.imshow('after', after)
    #cv2.imshow('diff', diff)
    #cv2.imshow('diff_box', diff_box)
    #cv2.imshow('mask', mask)
    #cv2.imshow('filled after', filled_after)
    #cv2.waitKey()
    value=[per,j,y]
    return value

def attack3(img1,img2):
    # Load images
    before = cv2.imread("static/upload/"+img1)
    after = cv2.imread("static/upload/"+img2)

    # Convert images to grayscale
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between the two images
    (score, diff) = structural_similarity(before_gray, after_gray, full=True)
    print("Image Similarity: {:.4f}%".format(score * 100))
    per=format(score * 100)

    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1] 
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] before we can use it with OpenCV
    diff = (diff * 255).astype("uint8")
    diff_box = cv2.merge([diff, diff, diff])

    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    mask = np.zeros(before.shape, dtype='uint8')
    filled_after = after.copy()
    j=1
    for c in contours:
        area = cv2.contourArea(c)
        if area > 40:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
            mm=cv2.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.imwrite("static/test/ggg.jpg", mm)

            image = cv2.imread("static/test/ggg.jpg")
            cropped = image[y:y+h, x:x+w]
            gg="g"+str(j)+".jpg"
            cv2.imwrite("static/test/"+gg, cropped)
        
            cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.drawContours(mask, [c], 0, (255,255,255), -1)
            cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)
            j+=1

    #########
    e=j-1
    n=1
    y=0
    k=0
    while n<=e:
        main_image = cv2.imread('static/upload/'+img2)
        gray_image = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
        #open the template as gray scale image
        ffn="g"+str(n)+".jpg"
        template = cv2.imread("static/test/"+ffn, 0)
        width, height = template.shape[::-1] #get the width and height
        #match the template using cv2.matchTemplate
        match = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        position = np.where(match >= threshold) #get the location of template in the image
        k=0
        for point in zip(*position[::-1]): #draw the rectangle around the matched template
           cv2.rectangle(main_image, point, (point[0] + width, point[1] + height), (0, 204, 153), 0)
           k+=1
        if k>1:
            y+=k
            
        n+=1
    ########
    #cv2.imshow('before', before)
    #cv2.imshow('after', after)
    #cv2.imshow('diff', diff)
    #cv2.imshow('diff_box', diff_box)
    #cv2.imshow('mask', mask)
    #cv2.imshow('filled after', filled_after)
    #cv2.waitKey()
    value=[per,j,y]
    return value

def vaccinate(uname,file_name):
    s=""
    user=""
    attack=""
    uu=""
    pid=""
    stu='0'
    fn=""
    social_app=0
    filepath="static/upload/"+file_name
    img1 = Image.open(filepath) 
    st="1"
    # get width and height 
    width1 = img1.width 
    height1 = img1.height
    
    mycursor = mydb.cursor()

    mycursor.execute("SELECT count(*) FROM im_post where photo!='' && status=0 order by id")
    cnt = mycursor.fetchone()[0]
    if cnt>0:
        mycursor.execute("SELECT * FROM im_post where photo!='' && status=0 order by id")
        dat = mycursor.fetchall()
        x=0
        for dd in dat:
            fn=dd[3]
            img=Image.open("static/upload/"+fn)
            width = img.width 
            height = img.height
            if width==width1 and height==height1:
                cmp=compare(fn,file_name)
                print(cmp)
                cmp1=float(cmp[0])
                uu=dd[1]
                if cmp1>=99.8:
                    x+=1
                    social_app=1
                    pid=str(dd[0])
                    if uname==dd[1]:
                        user="1"
                    else:
                        user="2"
                        
                        stu='1'
                    s="2"
                    break
                elif cmp1>=75:
                    social_app=1
                    pid=str(dd[0])
                    print(fn)
                    stu='2'
                    ff=open("static/fname.txt","w")
                    ff.write(fn)
                    ff.close()
                    x+=1
                    s="3"
                    st="2"
                    stu='2'
                    a1=attack1(fn,file_name)
                    a11=attack11(fn,file_name)
                    a2=attack2(fn,file_name)
                    attack_st2=a2[2]

                    a3=attack3(fn,file_name)
                    print("attacks1")
                    print(a1)
                    print("attacks11")
                    print(a11)
                    print("attacks2")
                    print(a2)
                    print("attacks3")
                    print(a3)

                    if a3[2]>0 and a2[2]==0 and a11[2]==0:
                        print("splice")
                        attack="splice"
                    elif a1[2]>a11[2] and a2[2]==0:
                        print("inpaint")
                        attack="inpaint"
                    elif a2[2]>a1[2] and a2[2]>a11[2]:
                        if a2[2]>0:
                            print("copy")
                            attack="copy"
                    elif a2[2]==0:
                        if a11[2]>=0:
                            if a1[2]>a11[2]:
                                print("inpaint")
                                attack="inpaint"
                    
                    



                    break
        ######
        mycursor.execute("SELECT * FROM im_post1 where photo!='' && status=0 order by id")
        dat1 = mycursor.fetchall()
        
        for dd1 in dat1:
            fn=dd1[3]
            img=Image.open("static/upload/"+fn)
            width = img.width 
            height = img.height
            if width==width1 and height==height1:
                cmp=compare(fn,file_name)
                print(cmp)
                cmp1=float(cmp[0])
                uu=dd1[1]
                if cmp1>=99.8:
                    x+=1
                    social_app=2
                    pid=str(dd1[0])
                    if uname==dd1[1]:
                        user="1"
                    else:
                        user="2"
                        #uu=dd1[1]
                        stu='1'
                    s="2"
                    break
                elif cmp1>=75:
                    social_app=2
                    pid=str(dd1[0])
                    
                    print(fn)
                    ff=open("static/fname.txt","w")
                    ff.write(fn)
                    ff.close()
                    x+=1
                    s="3"
                    st="2"
                    stu='2'
                    a1=attack1(fn,file_name)
                    a11=attack11(fn,file_name)
                    a2=attack2(fn,file_name)
                    attack_st2=a2[2]

                    a3=attack3(fn,file_name)
                    print("attacks1")
                    print(a1)
                    print("attacks11")
                    print(a11)
                    print("attacks2")
                    print(a2)
                    print("attacks3")
                    print(a3)

                    if a3[2]>0 and a2[2]==0 and a11[2]==0:
                        print("splice")
                        attack="splice"
                    elif a1[2]>a11[2] and a2[2]==0:
                        print("inpaint")
                        attack="inpaint"
                    elif a2[2]>a1[2] and a2[2]>a11[2]:
                        if a2[2]>0:
                            print("copy")
                            attack="copy"
                    elif a2[2]==0:
                        if a11[2]>=0:
                            if a1[2]>a11[2]:
                                print("inpaint")
                                attack="inpaint"

                    break
        ######
        if x==0:                    
            
            s="1"                    
            fn2="D"+file_name
            fn3="H"+file_name
            from PIL.ImageFilter import (
               BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
               EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN
            )
            #Create image object
            img = Image.open("static/upload/"+file_name)
            #Applying the blur filter
            img1 = img.filter(CONTOUR)
            img1.save('static/test/'+fn2)
            #img1.show()
            im = Image.open("static/test/D"+file_name)

            cmyk = gcr(im, 0)
            dots = halftone(im, cmyk, 10, 1)
            #im.show()
            new = Image.merge('CMYK', dots)
            #new.show()
            new.save('static/test/'+fn3)
           
    else:
        s="1"
        fn2="D"+file_name
        fn3="H"+file_name
        from PIL.ImageFilter import (
           BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
           EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN
        )
        #Create image object
        img = Image.open("static/upload/"+file_name)
        #Applying the blur filter
        img1 = img.filter(CONTOUR)
        img1.save('static/test/'+fn2)
        #img1.show()
        im = Image.open("static/test/D"+file_name)

        cmyk = gcr(im, 0)
        dots = halftone(im, cmyk, 10, 1)
        #im.show()
        new = Image.merge('CMYK', dots)
        #new.show()
        new.save('static/test/'+fn3)

    value=[s,user,uu,fn,pid,stu,attack,social_app]
    return value


@app.route('/userhome', methods=['GET', 'POST'])
def userhome():
    msg=""
    fid=""
    res=[]
    user=""
    st=""
    stu='0'
    pid=""
    app=0
    attack=""
    uu=""
    fnn=""
    file_name=""
    data2=[]
    req_cnt=0
    noti=""
    uname=""
    act=request.args.get("act")
    if 'username' in session:
        uname = session['username']
    
    mycursor = mydb.cursor()
    mycursor.execute('SELECT * FROM im_user where uname=%s',(uname,))
    udata = mycursor.fetchone()
   
    
    if request.method == 'POST':
        detail= request.form['detail']
        if 'file' not in request.files:
            flash('No file Part')
            return redirect(request.url)
        file= request.files['file']

        ##############
        mycursor.execute("SELECT count(*) FROM im_tamper")
        tcount = mycursor.fetchone()[0]
        ###
        if tcount>0:
            mycursor.execute("delete from im_tamper")
            mydb.commit()
        ##############
        
        mycursor.execute("SELECT max(id)+1 FROM im_post")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
        
        if file.filename == '':
            flash('No Select file')
            #return redirect(request.url)
        if file:
            fname1 = file.filename
            fname = secure_filename(fname1)
            file_name="P"+str(maxid)+fname
            file.save(os.path.join("static/upload/", file_name))
            ff=open("static/ufile.txt","w")
            ff.write(file_name)
            ff.close()
            res=vaccinate(uname,file_name)
            
            stu=res[5]
            pid=res[4]
            user=res[1]
            app=res[7]
            uu=res[2]
            fnn=res[3]
            ff=open("static/pid.txt","w")
            ff.write(pid)
            ff.close()
            
            ##
            '''fn2="D"+file_name
            fn3="H"+file_name
            from PIL.ImageFilter import (
               BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
               EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN
            )
            #Create image object
            img = Image.open("static/upload/"+file_name)
            #Applying the blur filter
            img1 = img.filter(CONTOUR)
            img1.save('static/upload/'+fn2)
            #img1.show()
            im = Image.open("static/upload/D"+file_name)

            cmyk = gcr(im, 0)
            dots = halftone(im, cmyk, 10, 1)
            #im.show()
            new = Image.merge('CMYK', dots)
            #new.show()
            new.save('static/upload/'+fn3)
            ##
            immunizer(uname,file_name)'''
            ##
            

        else:
            file_name=""
            
        today = date.today()
        rdate = today.strftime("%d-%m-%Y")

        rst=0
        if stu=='1' or stu=='2':
            rst=0
        else:
            rst=1

        
        
        sql = "INSERT INTO im_post (id,uname,detail,photo,rdate,status,post_id,post_user,social_app,request_st) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        val = (maxid,uname,detail,file_name,rdate,stu,pid,uu,app,rst)
        mycursor.execute(sql,val)
        print(sql,val)
        mydb.commit()

        if stu=='1' or stu=='2':
            if stu=="1":
                attack="similar"
            else:
                attack=res[6]
            mycursor.execute("SELECT max(id)+1 FROM im_tamper")
            maxid2 = mycursor.fetchone()[0]
            if maxid2 is None:
                maxid2=1

            

                
            sql = "INSERT INTO im_tamper (id,posted_user,attacked_user,original_id,post_id,original_img,attack_img,attack,request_status,social_app) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            val = (maxid2,uu,uname,pid,maxid,fnn,file_name,attack,'0','1')
            mycursor.execute(sql,val)
            print(sql,val)
            mydb.commit()
        if file_name=="":
            msg="success"
        else:
            fid=str(maxid)
            if res[0]=="1":
                msg="ok"
            elif res[0]=="2":
                if res[1]=="1":
                    msg="yes"
                else:
                    
                    msg="yes1"
                    
                
            elif res[0]=="3":
                attack=res[6]
                ff=open("static/attack.txt","w")
                ff.write(attack)
                ff.close()
                msg="attack"
        

    mycursor.execute('SELECT * FROM im_post where request_st=1 order by id desc')
    pdata = mycursor.fetchall()

    mycursor.execute('SELECT count(*) FROM im_post where uname=%s && request_st=1',(uname,))
    pcount = mycursor.fetchone()[0]

    ####
    mycursor.execute('SELECT count(*) FROM im_tamper where posted_user=%s && request_status=0',(uname,))
    cnt = mycursor.fetchone()[0]
    if cnt>0:
        req_cnt=cnt
        noti="1"
        mycursor.execute('SELECT * FROM im_tamper where posted_user=%s && request_status=0',(uname,))
        data2 = mycursor.fetchall()

    if act=="req1":
        rid=request.args.get("rid")
        mycursor.execute('SELECT * FROM im_tamper where id=%s',(rid,))
        ds1 = mycursor.fetchone()
        if ds1[9]==1:
            mycursor.execute("update im_post set request_st=1 where id=%s",(ds1[4],))
            mydb.commit()
        else:
            mycursor.execute("update im_post1 set request_st=1 where id=%s",(ds1[4],))
            mydb.commit()
        
        mycursor.execute("update im_tamper set request_status=1 where id=%s",(rid,))
        mydb.commit()
        return redirect(url_for('userhome'))
    if act=="req2":
        rid=request.args.get("rid")
        mycursor.execute('SELECT * FROM im_tamper where id=%s',(rid,))
        ds1 = mycursor.fetchone()
        if ds1[9]==1:
            mycursor.execute("delete from im_post where id=%s",(ds1[4],))
            mydb.commit()
        else:
            mycursor.execute("delete from im_post1 where id=%s",(ds1[4],))
            mydb.commit()
        mycursor.execute("update im_tamper set request_status=2 where id=%s",(rid,))
        mydb.commit()
        return redirect(url_for('userhome'))
    ####
        
    return render_template('userhome.html',msg=msg,udata=udata,pdata=pdata,pcount=pcount,fid=fid,user=uu,app=app,req_cnt=req_cnt,noti=noti,data2=data2)

@app.route('/show', methods=['GET', 'POST'])
def show():
    msg=""
    act=request.args.get("act")
    fid=request.args.get("fid")
    user=request.args.get("user")
    app=request.args.get("app")
    uname=""
    if 'username' in session:
        uname = session['username']
    
    mycursor = mydb.cursor()
    mycursor.execute('SELECT * FROM im_user where uname=%s',(uname,))
    udata = mycursor.fetchone()

    mycursor.execute('SELECT * FROM im_post where uname=%s && id=%s',(uname,fid))
    pdata = mycursor.fetchone()

    mycursor.execute('SELECT count(*) FROM im_post where uname=%s',(uname,))
    pcount = mycursor.fetchone()[0]

    return render_template('show.html',msg=msg,udata=udata,pdata=pdata,pcount=pcount,fid=fid,act=act,user=user,app=app)

@app.route('/show2', methods=['GET', 'POST'])
def show2():
    msg=""
    act=request.args.get("act")
    fid=request.args.get("fid")
    user=request.args.get("user")
    app=request.args.get("app")
    uname=""
    mess=""
    mess2=""
    if 'username' in session:
        uname = session['username']
    
    mycursor = mydb.cursor()
    mycursor.execute('SELECT * FROM im_user where uname=%s',(uname,))
    udata = mycursor.fetchone()
    email=udata[5]

    mycursor.execute('SELECT * FROM im_post where uname=%s && id=%s',(uname,fid))
    pdata = mycursor.fetchone()
    
    u2=pdata[8]
    if u2=="":
        s=1
    else:
        mycursor.execute('SELECT * FROM im_user where uname=%s',(u2,))
        udata2 = mycursor.fetchone()
        email2=udata2[5]
        mess="Posted image has similar to "+u2+" post, so wait for permission"
        mess2="Similar image has posted by "+uname
    

    mycursor.execute('SELECT count(*) FROM im_post where uname=%s',(uname,))
    pcount = mycursor.fetchone()[0]

    return render_template('show2.html',msg=msg,udata=udata,pdata=pdata,pcount=pcount,fid=fid,act=act,user=user,app=app,mess=mess,mess2=mess2,email=email,email2=email2)

@app.route('/show3', methods=['GET', 'POST'])
def show3():
    msg=""
    act=request.args.get("act")
    fid=request.args.get("fid")
    user=request.args.get("user")
    app=request.args.get("app")
    pdata1=[]
    attack=""
    uname=""
    u2=""
    if 'username' in session:
        uname = session['username']

    ff=open("static/attack.txt","r")
    a=ff.read()
    ff.close()
    if a=="copy":
        attack="Copy-move"
    elif a=="splice":
        attack="Splicing"
    else:
        attack="Inpainting"

    ff=open("static/ufile.txt","r")
    ufile=ff.read()
    ff.close()

    ff=open("static/pid.txt","r")
    pid=ff.read()
    ff.close()
        
    mycursor = mydb.cursor()
    mycursor.execute('SELECT * FROM im_user where uname=%s',(uname,))
    udata = mycursor.fetchone()

    mycursor.execute('SELECT * FROM im_post where uname=%s && id=%s',(uname,fid))
    pdata = mycursor.fetchone()
    if app==1:
        mycursor.execute('SELECT * FROM im_post where id=%s',(pid,))
        pdata1 = mycursor.fetchone()
    else:
        mycursor.execute('SELECT * FROM im_post1 where id=%s',(pid,))
        pdata1 = mycursor.fetchone()

    fname=pdata1[3]

    mycursor.execute('SELECT count(*) FROM im_post where uname=%s',(uname,))
    pcount = mycursor.fetchone()[0]

    return render_template('show3.html',msg=msg,udata=udata,pdata=pdata,pcount=pcount,fid=fid,act=act,user=user,attack=attack,fname=fname,ufile=ufile,app=app)

@app.route('/show31', methods=['GET', 'POST'])
def show31():
    msg=""
    act=request.args.get("act")
    fid=request.args.get("fid")
    user=request.args.get("user")
    app=request.args.get("app")
    pdata1=[]
    attack=""
    uname=""
    email=""
    email2=""
    u2=""
    if 'username' in session:
        uname = session['username']

    ff=open("static/attack.txt","r")
    a=ff.read()
    ff.close()
    if a=="copy":
        attack="Copy-move"
    elif a=="splice":
        attack="Splicing"
    else:
        attack="Inpainting"

    ff=open("static/ufile.txt","r")
    ufile=ff.read()
    ff.close()

    ff=open("static/pid.txt","r")
    pid=ff.read()
    ff.close()
        
    mycursor = mydb.cursor()
    mycursor.execute('SELECT * FROM im_user where uname=%s',(uname,))
    udata = mycursor.fetchone()
    email=udata[5]

    mycursor.execute('SELECT * FROM im_post where uname=%s && id=%s',(uname,fid))
    pdata = mycursor.fetchone()

    mycursor.execute('SELECT * FROM im_tamper where attacked_user=%s && post_id=%s',(uname,fid))
    pdata2 = mycursor.fetchone()
    u2=pdata2[1]
    #u2=pdata[8]
    if u2=="":
        s=1
    else:
        mycursor.execute('SELECT * FROM im_user1 where uname=%s',(u2,))
        udata2 = mycursor.fetchone()
        email2=udata2[5]
        mess="Attacked image has posted, so you will get permission from "+u2
        mess2="User: "+uname+" has posted by attacked image"
    
    '''if app==1:
        mycursor.execute('SELECT * FROM im_post where id=%s',(pid,))
        pdata1 = mycursor.fetchone()
    else:
        mycursor.execute('SELECT * FROM im_post1 where id=%s',(pid,))
        pdata1 = mycursor.fetchone()

    fname=pdata1[3]'''
    ff=open("static/fname.txt","r")
    fname=ff.read()
    ff.close()

    mycursor.execute('SELECT count(*) FROM im_post where uname=%s',(uname,))
    pcount = mycursor.fetchone()[0]

    return render_template('show31.html',msg=msg,udata=udata,pdata=pdata,pcount=pcount,fid=fid,act=act,user=user,attack=attack,fname=fname,ufile=ufile,app=app,mess=mess,mess2=mess2,email=email,email2=email2)

@app.route('/userhome1', methods=['GET', 'POST'])
def userhome1():
    msg=""
    act=request.args.get("act")
    fid=""
    res=[]
    user=""
    st=""
    stu='0'
    pid=""
    app=0
    attack=""
    uu=""
    fnn=""
    data2=[]
    req_cnt=0
    noti=""
    file_name=""
    uname=""
    if 'username' in session:
        uname = session['username']
    
    mycursor = mydb.cursor()
    mycursor.execute('SELECT * FROM im_user1 where uname=%s',(uname,))
    udata = mycursor.fetchone()
   
    
    if request.method == 'POST':
        detail= request.form['detail']
        if 'file' not in request.files:
            flash('No file Part')
            return redirect(request.url)
        file= request.files['file']

        ##############
        mycursor.execute("SELECT count(*) FROM im_tamper")
        tcount = mycursor.fetchone()[0]
        ###
        if tcount>0:
            mycursor.execute("delete from im_tamper")
            mydb.commit()
        ##############
            
        
        mycursor.execute("SELECT max(id)+1 FROM im_post1")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
        
        if file.filename == '':
            flash('No Select file')
            #return redirect(request.url)
        if file:
            fname1 = file.filename
            fname = secure_filename(fname1)
            file_name="R"+str(maxid)+fname
            file.save(os.path.join("static/upload/", file_name))
            ff=open("static/ufile.txt","w")
            ff.write(file_name)
            ff.close()
            res=vaccinate(uname,file_name)
            
            stu=res[5]
            pid=res[4]
            user=res[1]
            app=res[7]
            uu=res[2]
            fnn=res[3]
            ff=open("static/pid.txt","w")
            ff.write(pid)
            ff.close()
            ##
            '''fn2="D"+file_name
            fn3="H"+file_name
            from PIL.ImageFilter import (
               BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
               EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN
            )
            #Create image object
            img = Image.open("static/upload/"+file_name)
            #Applying the blur filter
            img1 = img.filter(CONTOUR)
            img1.save('static/upload/'+fn2)
            #img1.show()
            im = Image.open("static/upload/D"+file_name)

            cmyk = gcr(im, 0)
            dots = halftone(im, cmyk, 10, 1)
            #im.show()
            new = Image.merge('CMYK', dots)
            #new.show()
            new.save('static/upload/'+fn3)
            ##
            immunizer(uname,file_name)'''
            ##
            

        else:
            file_name=""
            
        today = date.today()
        rdate = today.strftime("%d-%m-%Y")

        rst=0
        if stu=='1' or stu=='2':
            rst=0
        else:
            rst=1

        
        
        sql = "INSERT INTO im_post1 (id,uname,detail,photo,rdate,status,post_id,post_user,social_app,request_st) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        val = (maxid,uname,detail,file_name,rdate,stu,pid,uu,app,rst)
        mycursor.execute(sql,val)
        print(sql,val)
        mydb.commit()

        if stu=='1' or stu=='2':
            if stu=="1":
                attack="similar"
            else:
                attack=res[6]
            mycursor.execute("SELECT max(id)+1 FROM im_tamper")
            maxid2 = mycursor.fetchone()[0]
            if maxid2 is None:
                maxid2=1
            
            sql = "INSERT INTO im_tamper (id,posted_user,attacked_user,original_id,post_id,original_img,attack_img,attack,request_status,social_app) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            val = (maxid2,uu,uname,pid,maxid,fnn,file_name,attack,'0','2')
            mycursor.execute(sql,val)
            print(sql,val)
            mydb.commit()
        
            
        if file_name=="":
            msg="success"
        else:
            fid=str(maxid)
            if res[0]=="1":
                msg="ok"
            elif res[0]=="2":
                if res[1]=="1":
                    msg="yes"
                else:
                    
                    msg="yes1"
                    
                
            elif res[0]=="3":
                attack=res[6]
                ff=open("static/attack.txt","w")
                ff.write(attack)
                ff.close()
                msg="attack"
        

    mycursor.execute('SELECT * FROM im_post1 where request_st=1 order by id desc')
    pdata = mycursor.fetchall()

    mycursor.execute('SELECT count(*) FROM im_post1 where uname=%s',(uname,))
    pcount = mycursor.fetchone()[0]

    ####
    mycursor.execute('SELECT count(*) FROM im_tamper where posted_user=%s && request_status=0',(uname,))
    cnt = mycursor.fetchone()[0]
    if cnt>0:
        req_cnt=cnt
        noti="1"
        mycursor.execute('SELECT * FROM im_tamper where posted_user=%s && request_status=0',(uname,))
        data2 = mycursor.fetchall()

    if act=="req1":
        rid=request.args.get("rid")
        mycursor.execute('SELECT * FROM im_tamper where id=%s',(rid,))
        ds1 = mycursor.fetchone()
        if ds1[9]==1:
            mycursor.execute("update im_post set request_st=1 where id=%s",(ds1[4],))
            mydb.commit()
        else:
            mycursor.execute("update im_post1 set request_st=1 where id=%s",(ds1[4],))
            mydb.commit()
        
        mycursor.execute("update im_tamper set request_status=1 where id=%s",(rid,))
        mydb.commit()
        return redirect(url_for('userhome1'))
    if act=="req2":
        rid=request.args.get("rid")
        mycursor.execute('SELECT * FROM im_tamper where id=%s',(rid,))
        ds1 = mycursor.fetchone()
        if ds1[9]==1:
            mycursor.execute("delete from im_post where id=%s",(ds1[4],))
            mydb.commit()
        else:
            mycursor.execute("delete from im_post1 where id=%s",(ds1[4],))
            mydb.commit()
        mycursor.execute("update im_tamper set request_status=2 where id=%s",(rid,))
        mydb.commit()
        return redirect(url_for('userhome1'))
    ####
        
    return render_template('web/userhome1.html',msg=msg,udata=udata,pdata=pdata,pcount=pcount,fid=fid,user=uu,app=app,req_cnt=req_cnt,noti=noti,data2=data2)

@app.route('/page', methods=['GET', 'POST'])
def page():
    msg=""
    act=request.args.get("act")
    fid=request.args.get("fid")
    user=request.args.get("user")
    app=request.args.get("app")
    uname=""
    if 'username' in session:
        uname = session['username']
    
    mycursor = mydb.cursor()
    mycursor.execute('SELECT * FROM im_user1 where uname=%s',(uname,))
    udata = mycursor.fetchone()

    mycursor.execute('SELECT * FROM im_post1 where uname=%s && id=%s',(uname,fid))
    pdata = mycursor.fetchone()

    mycursor.execute('SELECT count(*) FROM im_post1 where uname=%s',(uname,))
    pcount = mycursor.fetchone()[0]

    return render_template('web/page.html',msg=msg,udata=udata,pdata=pdata,pcount=pcount,fid=fid,act=act,user=user,app=app)

@app.route('/page2', methods=['GET', 'POST'])
def page2():
    msg=""
    act=request.args.get("act")
    fid=request.args.get("fid")
    user=request.args.get("user")
    app=request.args.get("app")
    uname=""
    email2=""
    email=""
    mess=""
    mess2=""
    if 'username' in session:
        uname = session['username']
    
    mycursor = mydb.cursor()
    uname="ganesh"
    mycursor.execute('SELECT * FROM im_user1 where uname=%s',(uname,))
    udata = mycursor.fetchone()
    email=udata[5]
    print(uname)
    mycursor.execute('SELECT * FROM im_post1 where uname=%s && id=%s',(uname,fid))
    pdata = mycursor.fetchone()

    u2=pdata[8]
    if u2=="":
        s=1
    else:
        mycursor.execute('SELECT * FROM im_user where uname=%s',(u2,))
        udata2 = mycursor.fetchone()
        email2=udata2[5]
        mess="Posted image has similar to "+u2+" post, so wait for permission"
        mess2="Similar image has posted by "+uname
    

    mycursor.execute('SELECT count(*) FROM im_post1 where uname=%s',(uname,))
    pcount = mycursor.fetchone()[0]

    return render_template('web/page2.html',msg=msg,udata=udata,pdata=pdata,pcount=pcount,fid=fid,act=act,user=user,app=app,mess=mess,mess2=mess2,email=email,email2=email2)

@app.route('/page3', methods=['GET', 'POST'])
def page3():
    msg=""
    act=request.args.get("act")
    fid=request.args.get("fid")
    user=request.args.get("user")
    app=request.args.get("app")
    pdata1=[]
    attack=""
    uname=""
    fname=""
    if 'username' in session:
        uname = session['username']

    
    

    ff=open("static/attack.txt","r")
    a=ff.read()
    ff.close()
    if a=="copy":
        attack="Copy-move"
    elif a=="splice":
        attack="Splicing"
    else:
        attack="Inpainting"
    print(attack)

    ff=open("static/ufile.txt","r")
    ufile=ff.read()
    ff.close()

    ff=open("static/pid.txt","r")
    pid=ff.read()
    ff.close()
        
    mycursor = mydb.cursor()
    mycursor.execute('SELECT * FROM im_user1 where uname=%s',(uname,))
    udata = mycursor.fetchone()

    mycursor.execute('SELECT * FROM im_post1 where uname=%s && id=%s',(uname,fid))
    pdata = mycursor.fetchone()

    

    if app==1:
        mycursor.execute('SELECT * FROM im_post where id=%s',(pid,))
        pdata1 = mycursor.fetchone()
    else:
        mycursor.execute('SELECT * FROM im_post1 where id=%s',(pid,))
        pdata1 = mycursor.fetchone()
    #fname=pdata1[3]
    

    mycursor.execute('SELECT count(*) FROM im_post1 where uname=%s',(uname,))
    pcount = mycursor.fetchone()[0]

    return render_template('web/page3.html',msg=msg,udata=udata,pdata=pdata,pcount=pcount,fid=fid,act=act,user=user,attack=attack,fname=fname,ufile=ufile,app=app)

@app.route('/page31', methods=['GET', 'POST'])
def page31():
    msg=""
    act=request.args.get("act")
    fid=request.args.get("fid")
    user=request.args.get("user")
    app=request.args.get("app")
    pdata1=[]
    p2=""
    attack=""
    email=""
    email2=""
    mess=""
    mess2=""
    fname=""
    u2=""
    uname=""
    if 'username' in session:
        uname = session['username']

    ff=open("static/attack.txt","r")
    a=ff.read()
    ff.close()
    if a=="copy":
        attack="Copy-move"
    elif a=="splice":
        attack="Splicing"
    else:
        attack="Inpainting"
    print(attack)

    ff=open("static/ufile.txt","r")
    ufile=ff.read()
    ff.close()

    ff=open("static/pid.txt","r")
    pid=ff.read()
    ff.close()
        
    mycursor = mydb.cursor()
    mycursor.execute('SELECT * FROM im_user1 where uname=%s',(uname,))
    udata = mycursor.fetchone()
    email=udata[5]

    mycursor.execute('SELECT * FROM im_post1 where uname=%s && id=%s',(uname,fid))
    pdata = mycursor.fetchone()
    
    mycursor.execute('SELECT * FROM im_tamper where attacked_user=%s && post_id=%s',(uname,fid))
    pdata2 = mycursor.fetchone()
    u2=pdata2[1]
    #u2=pdata[8]
    if u2=="":
        s=1
    else:
        mycursor.execute('SELECT * FROM im_user where uname=%s',(u2,))
        udata2 = mycursor.fetchone()
        email2=udata2[5]
        mess="Attacked image has posted, so you will get permission from "+u2
        mess2="User: "+uname+" has posted by attacked image"
    

    '''if app==1:
        mycursor.execute('SELECT * FROM im_post where id=%s',(pid,))
        pdata1 = mycursor.fetchone()
        fname=pdata1[3]
    else:
        mycursor.execute('SELECT * FROM im_post1 where id=%s',(pid,))
        pdata1 = mycursor.fetchone()
        fname=pdata1[3]'''

    ff=open("static/fname.txt","r")
    fname=ff.read()
    ff.close()
                    
    print("filename")
    print(fname)
    
    mycursor.execute('SELECT count(*) FROM im_post1 where uname=%s',(uname,))
    pcount = mycursor.fetchone()[0]

    return render_template('web/page31.html',msg=msg,udata=udata,pdata=pdata,pcount=pcount,fid=fid,act=act,user=user,attack=attack,fname=fname,ufile=ufile,app=app,mess=mess,mess2=mess2,email=email,email2=email2)

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    msg=""
    uname=""
    if 'username' in session:
        uname = session['username']
    
    mycursor = mydb.cursor()
    mycursor.execute('SELECT * FROM im_tamper order by id desc')
    data = mycursor.fetchall()
        

    return render_template('admin.html',msg=msg,data=data)

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

@app.route('/admin_view', methods=['GET', 'POST'])
def admin_view():
    msg=""
    mid=request.args.get("mid")
    uname=""
    if 'username' in session:
        uname = session['username']
    
    mycursor = mydb.cursor()
    mycursor.execute('SELECT * FROM im_tamper where id=%s',(mid,))
    data = mycursor.fetchone()

    img1=data[5]
    img2=data[6]
    #########
    before = cv2.imread("static/upload/"+img2)
    after = cv2.imread("static/upload/"+img1)

    # Convert images to grayscale
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between the two images
    (score, diff) = structural_similarity(before_gray, after_gray, full=True)
    print("Image Similarity: {:.4f}%".format(score * 100))
    per=format(score * 100)

    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1] 
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] before we can use it with OpenCV
    diff = (diff * 255).astype("uint8")
    diff_box = cv2.merge([diff, diff, diff])

    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    mask = np.zeros(before.shape, dtype='uint8')
    filled_after = after.copy()
    j=1
    for c in contours:
        area = cv2.contourArea(c)
        if area > 40:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
            mm=cv2.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)
            #cv2.imwrite("static/test/ggg.jpg", mm)

            #image = cv2.imread("static/test/ggg.jpg")
            #cropped = image[y:y+h, x:x+w]
            #gg="u"+str(j)+".jpg"
            #cv2.imwrite("static/test/"+gg, cropped)
        
            cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.drawContours(mask, [c], 0, (255,255,255), -1)
            cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)
            j+=1

    cv2.imwrite("static/test/t1.jpg", before)
    cv2.imwrite("static/test/t2.jpg", after)
    cv2.imwrite("static/test/t3.jpg", diff)
    cv2.imwrite("static/test/t4.jpg", diff_box)
    cv2.imwrite("static/test/t5.jpg", mask)
    cv2.imwrite("static/test/t6.jpg", filled_after)
    ##############     
    im = cv2.imread('static/upload/'+img1)
    sh=im.shape
    
    histr = cv2.calcHist([im],[0],None,[256],[0,256])
    plt.plot(histr)
    himg="H1.png"
    plt.savefig("static/test/"+himg)
    plt.close()
    #########
    im = cv2.imread('static/upload/'+img2)
    sh=im.shape
    
    histr = cv2.calcHist([im],[0],None,[256],[0,256])
    plt.plot(histr)
    himg="H2.png"
    plt.savefig("static/test/"+himg)
    plt.close()
    ##
    ##############
    #pvalue=PSNR("static/upload/"+fn,"static/upload/"+comfile)
    original = cv2.imread("static/upload/"+img1)
    compressed = cv2.imread("static/upload/"+img2, 1)
    value = PSNR(original, compressed)
    ps="PSNR value is "+str(value)+" dB"
    print(ps)
    psnr=round(value,2)
    
    return render_template('admin_view.html',msg=msg,data=data,psnr=psnr)


@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    session.pop('username', None)
    return redirect(url_for('index'))



if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(debug=True,host='0.0.0.0', port=5000)


