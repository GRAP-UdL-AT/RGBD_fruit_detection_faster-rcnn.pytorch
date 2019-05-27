from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import _init_paths
import os
import numpy as np
import pandas as pd
import shutil
import PIL
from PIL import Image
import pdb
from lxml import etree as ET
from model.utils.config import cfg

path = os.path.join(cfg.DATA_DIR,'kinect_fruits_dataset','apples','annotations')
Files = os.listdir(path)
for i in range(1,2+1):
	save_dir = os.path.join(cfg.DATA_DIR,'kinect_fruits_dataset','apples','square_annotations'+str(i)+'')
	print("Writing on folder: %s"%save_dir)
	img_dir = os.path.join(cfg.DATA_DIR,'kinect_fruits_dataset','apples','images')
	if os.path.exists(save_dir):
		shutil.rmtree(save_dir)
		os.makedirs(save_dir)
	else:
		os.makedirs(save_dir)


	for file in Files:
		full_filename = os.path.join(path,file)
		width = PIL.Image.open(os.path.join(img_dir,file.split(".csv")[0] + ".jpg")).size[0]
		height = PIL.Image.open(os.path.join(img_dir,file.split(".csv")[0] + ".jpg")).size[1]
		circles = pd.read_csv(full_filename, header=None)
		circles = circles.values
		#Square_annotations1
		if i == 1:	
			xmin = circles[:,1] 
			xmax = circles[:,1] + circles[:,3] 
			ymin= circles[:,2] 
			ymax = circles[:,2] + circles[:,4]
		#Square_annotations2
		if i == 2:
			xmin = circles[:,2] 
			xmax = circles[:,2] + circles[:,3] 
			ymin= circles[:,1] 
			ymax = circles[:,1] + circles[:,3]
		for index, item in enumerate(xmin):
			if item < 0:
				xmin[index] = 1	
		for index, item in enumerate(ymin):
			if item < 0:
				ymin[index] = 1
		for index, item in enumerate(xmax):
			if item > width:
				xmax[index] = width
		for index, item in enumerate(ymax):
			if item > height:
				ymax[index] = height
		
		f = ET.Element("annotations")
		ET.SubElement(f,'filename').text = file.split(".csv")[0] + ".jpg"
		thesize = ET.SubElement(f,'size')
		ET.SubElement(thesize,'width').text = str(width)
		ET.SubElement(thesize,'height').text = str(height)
		ET.SubElement(thesize,'depth').text = "3"
		for j in range(len(xmin)):

			obj = ET.SubElement(f,'object')
			ET.SubElement(obj,'name').text = "Poma"
			ET.SubElement(obj,'difficult').text = "0"
			bbox = ET.SubElement(obj,'bbox')
			xmin_xml = ET.SubElement(bbox,'xmin')		
			ymin_xml = ET.SubElement(bbox,'ymin')
			xmax_xml = ET.SubElement(bbox,'xmax')
			ymax_xml = ET.SubElement(bbox,'ymax')		
			xmin_xml.text = str(int(xmin[j]))
			xmax_xml.text = str(int(xmax[j]))
			ymin_xml.text = str(int(ymin[j]))
			ymax_xml.text = str(int(ymax[j]))
		filename = os.path.join(save_dir,file.split(".csv")[0] + ".xml")
		tree = ET.ElementTree(f)
		
		tree.write(filename, pretty_print = True)
	





	
