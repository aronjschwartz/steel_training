#Project: Steel defect machine learning for Kaggle Competition
#Last Edit: Aron Schwartz 8/10/2019

import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import time

# Root directory of the project
ROOT_DIR = os.path.abspath("C:/Users/aronj/Kaggle_Competition/steel_training/")
# Import Mask RCNN

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_WEIGHTS_PATH):
    utils.download_trained_weights(COCO_WEIGHTS_PATH)
	
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")	
		
class SteelConfig(Config):
    
	NAME = "steel"
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

    # Number of classes (including background)
	NUM_CLASSES = 4 #Four for now (may need class 0 later)
	
	#1600x256 image size
	IMAGE_MIN_DIM = 256
	IMAGE_MAX_DIM = 1600

    # Use smaller anchors because our image and objects are small
	RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128, )  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
	TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
	STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
	VALIDATION_STEPS = 5
	
	DETECTION_MIN_CONFIDENCE = 0.80

    # Non-maximum suppression threshold for detection
	DETECTION_NMS_THRESHOLD = 0.0
	
def isNaN(num):
    return num != num	
	
class SteelDataset(utils.Dataset):


	def load_steel(self, dataset_dir, subset):
	
		#self.add_class("Type_0", 0, "Type_0")
		self.add_class("Type_1", 1, "Type_1")
		self.add_class("Type_2", 2, "Type_2")
		self.add_class("Type_3", 3, "Type_3")
		self.add_class("Type_4", 4, "Type_4")

		assert subset in ["train", "val"]
		dataset_dir = os.path.join(dataset_dir, subset)
		dataset_dir = dataset_dir.replace("/", "\\")
		# Load image ids (filenames) and run length encoded pixels
		temp_str = str(os.path.join(dataset_dir, "{}.csv".format(subset)))
				
		#Extract the pandas dataframe from the data file
		steel_segmentations_df = pd.read_csv(os.path.join(dataset_dir, "{}.csv".format(subset)))
		
		#Dictionary to hold images and their masks
		images_mask_dict = {}	
		defect_type_dict = {}
		#Loop through each data frame and create a dict that has a image_id and the pixels of the mask (if exists)
		for index, row in steel_segmentations_df.iterrows():
			image_ident_temp = row['image_id']
			split = image_ident_temp.split("_")
			image_ident = split[0]
			if isNaN(row['EncodedPixels']) and image_ident not in images_mask_dict:
				images_mask_dict[image_ident] = row['EncodedPixels']	
				defect_type_dict[image_ident] = 0
			elif isNaN(row['EncodedPixels']) and image_ident in images_mask_dict:
				continue
			elif not isNaN(row['EncodedPixels']):
				images_mask_dict[image_ident] = row['EncodedPixels']
				defect_type_dict[image_ident] = int(split[1])
		
		
		for key, val in images_mask_dict.items():
				
			image_path = os.path.join(dataset_dir, key)
			#print("The image path is: ", image_path)
			if os.path.isfile(image_path):		
				
				if defect_type_dict[key] == 1:
					self.add_image(
                    "Type_1",
                    image_id=key,  # use file name as a unique image id
                    path=image_path,
                    width=1600, height=256,
                    img_masks=val)
								
				elif defect_type_dict[key] == 2:
					self.add_image(
                    "Type_2",
                    image_id=key,  # use file name as a unique image id
                    path=image_path,
                    width=1600, height=256,
                    img_masks=val)
							
				elif defect_type_dict[key] == 3:
					self.add_image(
                    "Type_3",
                    image_id=key,  # use file name as a unique image id
                    path=image_path,
                    width=1600, height=256,
                    img_masks=val)			
				
				elif defect_type_dict[key] == 4:
					self.add_image(
                    "Type_4",
                    image_id=key,  # use file name as a unique image id
                    path=image_path,
                    width=1600, height=256,
                    img_masks=val)
			
	def load_mask(self, image_id):
	
	
		image_info = self.image_info[image_id]
		if image_info["source"] != (("Type_1") or ("Type_2") or ("Type_3") or ("Type_4")):
			return super(self.__class__, self).load_mask(image_id)		
		# Convert RLE Encoding to bitmap mask of shape [height, width, instance count]
		temp_id = ""
		temp_mask = ""
		width = ""
		height = ""
		for key, val in image_info.items():
			if key == 'id':
				temp_id = val
			elif key == 'img_masks':
				temp_mask = val
			elif key == 'width':
				width = val
			elif key == 'height':
				height = val
		
		
		shape = [height, width]
		
		# Mask array placeholder
		mask_array = np.zeros([width, height, len(temp_mask)],dtype=np.uint8)

		# Build mask array
		for index, mask in enumerate(temp_mask):
			mask_array[:,:,index] = self.rle_decode(mask, shape)
		return mask_array.astype(np.bool), np.ones([mask_array.shape[-1]], dtype=np.int32)		
			


	def image_reference(self, image_id):
		"""Return the path of the image."""
		info = self.image_info[image_id]
		if info["source"] == "ship":
			return info["path"]
		else:
			super(self.__class__, self).image_reference(image_id)

	def rle_encode(self,img):
		'''
		img: numpy array, 1 - mask, 0 - background
		Returns run length as string formated
		'''
		pixels = img.T.flatten()
		pixels = np.concatenate([[0], pixels, [0]])
		runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
		runs[1::2] -= runs[::2]
		return ' '.join(str(x) for x in runs)

	def rle_decode(self, mask_rle, shape=(1600, 256)):
		'''
		mask_rle: run-length as string formated (start length)
		shape: (height,width) of array to return
		Returns numpy array, 1 - mask, 0 - background
		'''
		if not isinstance(mask_rle, str):
			img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
			return img.reshape(shape).T

		s = mask_rle.split()
		starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
		starts -= 1
		ends = starts + lengths
		img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
		for lo, hi in zip(starts, ends):
			img[lo:hi] = 1
		return img.reshape(shape).T

	def multi_rle_encode(self, mask_array):
		# Go back from Bitmask to RLE
		re_encoded_to_rle_list = []
		for i in np.arange(mask_array.shape[-1]):
			boolean_mask = mask_array[:,:,i]
			re_encoded_to_rle = self.rle_encode(boolean_mask)
			re_encoded_to_rle_list.append(re_encoded_to_rle)

		return re_encoded_to_rle_list

	def multi_rle_decode(self, rle_img_masks):
		# Build mask array
		mask_array = np.zeros([768, 768, len(rle_img_masks)],dtype=np.uint8)

		# Go from RLE to Bitmask
		for index, rle_mask in enumerate(rle_img_masks):
			mask_array[:,:,index] = self.rle_decode(rle_mask)

		return mask_array

	def test_endcode_decode(self):
		ROOT_DIR = os.path.abspath("../../")
		SHIP_DIR = os.path.join(ROOT_DIR, "./samples/ship/datasets")
		ship_segmentations_df = pd.read_csv(os.path.join(SHIP_DIR,"train_val","train_ship_segmentations.csv"))
		rle_img_masks = ship_segmentations_df.loc[ship_segmentations_df['image_id'] == "0005d01c8.jpg", 'EncodedPixels']
		rle_img_masks_list = rle_img_masks.tolist()

		mask_array = self.multi_rle_decode(rle_img_masks)
		print("mask_array shape", mask_array.shape)
		# re_encoded_to_rle_list = self.multi_rle_encode(mask_array)
		re_encoded_to_rle_list = []
		for i in np.arange(mask_array.shape[-1]):
			boolean_mask = mask_array[:,:,i]
			re_encoded_to_rle = self.rle_encode(boolean_mask)
			re_encoded_to_rle_list.append(re_encoded_to_rle)

		print("Masks Match?", re_encoded_to_rle_list == rle_img_masks_list)
		print("Mask Count: ", len(rle_img_masks))
		print("rle_img_masks_list", rle_img_masks_list)
		print("re_encoded_to_rle_list", re_encoded_to_rle_list)

		# Check if re encoded rle masks are the same as the original ones
		return re_encoded_to_rle_list == rle_img_masks_list

	def masks_as_image(self,in_mask_list):
		# Take the individual ship masks and create a single mask array for all ships
		all_masks = np.zeros((256, 1600), dtype = np.uint8)
		for mask in in_mask_list:
			if isinstance(mask, str):
				all_masks |= self.rle_decode(mask)
		return all_masks

	def multi_rle_encode(self,img):
		labels = label(img)
		if img.ndim > 2:
			return [self.rle_encode(np.sum(labels==k, axis=2)) for k in np.unique(labels[labels>0])]
		else:
			return [self.rle_encode(labels==k) for k in np.unique(labels[labels>0])]

	

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = SteelDataset()
    dataset_train.load_steel(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = SteelDataset()
    dataset_val.load_steel(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=2,
                layers='heads')	
	
	
	
	
def color_splash(image, mask):
	"""Apply color splash effect.
	image: RGB image [height, width, 3]
	mask: instance segmentation mask [height, width, instance count]
	Returns result image.
	"""
	# Make a grayscale copy of the image. The grayscale copy still
	# has 3 RGB channels, though.
	gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
	# Copy color pixels from the original color image where mask is set
	if mask.shape[-1] > 0:
		# We're treating all instances as one, so collapse the mask into one layer
		mask = (np.sum(mask, -1, keepdims=True) >= 1)
		splash = np.where(mask, image, gray).astype(np.uint8)
	else:
		splash = gray.astype(np.uint8)
	return splash

	
def detect_and_color_splash(model, image_path=None, video_path=None):
	assert image_path or video_path

	# Image or video?
	if image_path:
		# Run model detection and generate the color splash effect
		print("Running on {}".format(args.image))
		# Read image
		image = skimage.io.imread(args.image)
		# Detect objects
		r = model.detect([image], verbose=1)[0]
		# Color splash
		splash = color_splash(image, r['masks'])
		# Save output
		file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
		skimage.io.imsave(file_name, splash)
	elif video_path:
		import cv2
		# Video capture
		vcapture = cv2.VideoCapture(video_path)
		width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fps = vcapture.get(cv2.CAP_PROP_FPS)

		# Define codec and create video writer
		file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
		vwriter = cv2.VideoWriter(file_name,
								  cv2.VideoWriter_fourcc(*'MJPG'),
								  fps, (width, height))

		count = 0
		success = True
		while success:
			print("frame: ", count)
			# Read next image
			success, image = vcapture.read()
			if success:
				# OpenCV returns images as BGR, convert to RGB
				image = image[..., ::-1]
				# Detect objects
				r = model.detect([image], verbose=0)[0]
				# Color splash
				splash = color_splash(image, r['masks'])
				# RGB -> BGR to save image to video
				splash = splash[..., ::-1]
				# Add image to video writer
				vwriter.write(splash)
				count += 1
		vwriter.release()
	print("Saved to ", file_name)

	

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect ships.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/ship/dataset/",
                        help='Directory of the Ship dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
   
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = SteelConfig()
    else:
        class InferenceConfig(SteelConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))


