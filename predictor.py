# -*- coding: utf-8 -*-
"""
Dog Breed Predictor
"""

import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import applications

TARGET_SIZE = (224,224)

DOG_BREEDS = np.array(['Chihuahua', 'Japanese_spaniel', 'Maltese_dog', 'Pekinese', 'Shih_Tzu', 'Blenheim_spaniel', 
              'papillon', 'toy_terrier', 'Rhodesian_ridgeback', 'Afghan_hound', 'basset', 'beagle', 'bloodhound', 
              'bluetick', 'black_and_tan_coonhound', 'Walker_hound', 'English_foxhound', 'redbone', 'borzoi', 
              'Irish_wolfhound', 'Italian_greyhound', 'whippet', 'Ibizan_hound', 'Norwegian_elkhound', 'otterhound', 
              'Saluki', 'Scottish_deerhound', 'Weimaraner', 'Staffordshire_bullterrier', 'American_Staffordshire_terrier', 
              'Bedlington_terrier', 'Border_terrier', 'Kerry_blue_terrier', 'Irish_terrier', 'Norfolk_terrier', 
              'Norwich_terrier', 'Yorkshire_terrier', 'wire_haired_fox_terrier', 'Lakeland_terrier', 'Sealyham_terrier', 
              'Airedale', 'cairn', 'Australian_terrier', 'Dandie_Dinmont', 'Boston_bull', 'miniature_schnauzer', 
              'giant_schnauzer', 'standard_schnauzer', 'Scotch_terrier', 'Tibetan_terrier', 'silky_terrier', 
              'soft_coated_wheaten_terrier', 'West_Highland_white_terrier', 'Lhasa', 'flat_coated_retriever', 
              'curly_coated_retriever', 'golden_retriever', 'Labrador_retriever', 'Chesapeake_Bay_retriever', 
              'German_short_haired_pointer', 'vizsla', 'English_setter', 'Irish_setter', 'Gordon_setter', 
              'Brittany_spaniel', 'clumber', 'English_springer', 'Welsh_springer_spaniel', 'cocker_spaniel', 
              'Sussex_spaniel', 'Irish_water_spaniel', 'kuvasz', 'schipperke', 'groenendael', 'malinois', 'briard', 
              'kelpie', 'komondor', 'Old_English_sheepdog', 'Shetland_sheepdog', 'collie', 'Border_collie', 
              'Bouvier_des_Flandres', 'Rottweiler', 'German_shepherd', 'Doberman', 'miniature_pinscher', 
              'Greater_Swiss_Mountain_dog', 'Bernese_mountain_dog', 'Appenzeller', 'EntleBucher', 'boxer', 
              'bull_mastiff', 'Tibetan_mastiff', 'French_bulldog', 'Great_Dane', 'Saint_Bernard', 'Eskimo_dog', 
              'malamute', 'Siberian_husky', 'affenpinscher', 'basenji', 'pug', 'Leonberg', 'Newfoundland', 
              'Great_Pyrenees', 'Samoyed', 'Pomeranian', 'chow', 'keeshond', 'Brabancon_griffon', 'Pembroke', 
              'Cardigan', 'toy_poodle', 'miniature_poodle', 'standard_poodle', 'Mexican_hairless', 'dingo', 
              'dhole', 'African_hunting_dog'])


MODEL = load_model("xception_120_breeds_fine_tuned.h5")

def load_picture(img_path):
	img = load_img(img_path, target_size=TARGET_SIZE) 
	img_array = img_to_array(img, dtype='float16')
	return img_array

def preprocess(img_array):
	#apply model preprocess_input function
	processed_img = applications.xception.preprocess_input(img_array)
	#reshape it to 4 dimensions
	processed_img = np.reshape(processed_img, (1,*TARGET_SIZE,3))
	return processed_img

def predict_breed(img):
	#requires a picture that has already been processed
	pred = MODEL.predict(img)[0]
	return pred

def translate_predictions(pred):
	#returns top 5 probabilities, with breed names
	top_5 = np.argsort(pred)[-5:][::-1]
	probas = pred[top_5]
	breeds = DOG_BREEDS[top_5]
	return probas, breeds

def run_predictor(img_path):
	#all-in-one wrapper
	img_array = load_picture(img_path)
	processed_img = preprocess(img_array)
	preds = predict_breed(processed_img)
	probas, breeds = translate_predictions(preds)
	return probas, breeds
