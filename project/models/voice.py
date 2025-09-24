# -*- coding: utf-8 -*-
from flask import flash
import pyaudio
import sys
import wave
import cv2
import os
import glob
import itertools
import csv
from cryptography.fernet import Fernet
import pickle
import time
import io
import requests
import numpy as np
from pinatapy import PinataPy
from scipy.io.wavfile import read
from IPython.display import Audio, display, clear_output
from retry import retry

from project.models.main_functions import *

pinata = PinataPy("", "")

class voice(object):

    def __init__(self) -> None:
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 2
        self.RATE = 48000
        self.CHUNK = 1024
        self.RECORD_SECONDS = 3

        self.FILENAME = "./test.wav"
        self.MODEL = "\\gmm_models\\voice_auth.gmm"
        self.VOICEPATH = "\\voice_database\\"
        #Regularly updated GMM libraries IPFS CID
        self.IPFS_HASH = ""

        #Encryption keys for the GMM library
        self.IPFS_PRIVATE_KEY = Fernet.generate_key()
        self.fernet = Fernet(self.IPFS_PRIVATE_KEY)

        self.VOICEDICT = {}

    def pad_or_trim(self, vector, target_length=6000):
        if len(vector) > target_length:
            return vector[:target_length]
        elif len(vector) < target_length:
            return np.pad(vector, (0, target_length - len(vector)), mode='constant')
        return vector
    
    def mutate_bytes(self, wav_bytes: bytes, offset: int) -> bytes:
        byte_array = bytearray(wav_bytes)
        if offset < len(byte_array):
            byte_array[offset] = (byte_array[offset] + 1) % 256
        return bytes(byte_array)

    def add_user(self, voice1, voice2, voice3, username):   

        result = False
        source = self.VOICEPATH + username
        absolute_path = os.path.dirname(__file__) + self.VOICEPATH + username
        os.mkdir(absolute_path)
        
        voice_dir = [voice1, voice2, voice3]
        self.VOICEDICT[username] = voice_dir
        X = []
        Y = []
        for name in voice_dir:
            # reading audio files of speaker
            (sr, audio) = read(io.BytesIO(name))
                    
            # extract 40 dimensional MFCC
            vector = extract_features(audio,sr).flatten()
            vector = self.pad_or_trim(vector, target_length=6000)
            X.append(vector)
            Y.append(name)
        X = np.array(X, dtype=object)

        le = preprocessing.LabelEncoder()
        le.fit(Y)
        Y_trans = le.transform(Y)
        clf = LogisticRegression(random_state=0).fit(X.tolist(), Y_trans)

        if os.path.isfile(os.path.dirname(__file__) + "\\gmm_models\\voice_auth.gmm"): 
            os.remove(os.path.dirname(__file__) + "\\gmm_models\\voice_auth.gmm")
            if self.IPFS_HASH != "":
                response = pinata.remove_pin_from_ipfs(self.IPFS_HASH)   
                #print(response)  
        # saving model
        pickle.dump(clf, open(os.path.dirname(__file__) + '\\gmm_models\\voice_auth.gmm', 'wb'))
        try:
            #Encrypt Biometric Database
            with open(os.path.dirname(__file__) + "\\gmm_models\\voice_auth.gmm", "rb") as voice_dict:
                safe_voice_dict = open(os.path.dirname(__file__) + "\\gmm_models\\encrypted_voice_auth.gmm", "w")
                encrypted_voice = self.fernet.encrypt(voice_dict.read())
                safe_voice_dict.write(str(encrypted_voice, 'utf-8'))
                safe_voice_dict.close()
        except Exception as e:
            print("Error during encryption:", e)
        #Write Biometric Database to IPFS
        if os.path.isfile(os.path.dirname(__file__) + "\\gmm_models\\encrypted_voice_auth.gmm"):
            response = pinata.pin_file_to_ipfs(os.path.dirname(__file__) + '\\gmm_models\\encrypted_voice_auth.gmm')   
            #print(response)
            self.IPFS_HASH = response['IpfsHash']

            #print(username + ' added successfully') 
            result = True
        
        features = np.asarray(())
        return result
    
    
    def loadWAV(self, file_paths):
        wav_streams = []
        for path in os.listdir(file_paths):
            full_path = os.path.join(file_paths, path)
            try:
                # Check if it's a directory (like 'china')
                if os.path.isdir(full_path):
                    # Skip directories, only process files
                    continue
                # Check if it's a .wav file
                elif path.lower().endswith('.wav'):
                    with open(full_path, 'rb') as f:
                        wav_bytes = f.read()
                        filename = os.path.basename(path)
                        wav_streams.append((filename.split("_")[0].lower(), wav_bytes))
            except Exception as e:
                print(f"Error loading {path}: {e}")
        return wav_streams
    
    
    def test(self):
        data = self.loadWAV("project/models/test_wav/")
        emotion_map = ['amused', 'anger', 'disgust', 'neutral', 'sleepiness']
        assert len(data) == 5, "Expected exactly 5 emotional samples."

        emotion_data = {label: wav_bytes for label, wav_bytes in zip(emotion_map, [d[1] for d in data])}
        neutral_wav = emotion_data['neutral']
        username = "neutral_user3"
        results = []

        # Prepare three slightly different versions of the same neutral recording
        train_samples = [
                self.mutate_bytes(neutral_wav, offset=1000),
                self.mutate_bytes(neutral_wav, offset=2000),
                self.mutate_bytes(neutral_wav, offset=3000)
            ]

        # Train the model
        self.add_user(train_samples[0], train_samples[1], train_samples[2], username)

        for iteration in range(1, 301):

            # Test against all 5 emotions with retry logic
            for test_emotion in emotion_map:
                while True:
                    try:
                        result = self.recognise(emotion_data[test_emotion], username)
                        results.append({
                            "iteration": iteration,
                            "train_emotion": 'neutral',
                            "test_emotion": test_emotion,
                            "match_score": round(result[1], 2) if isinstance(result[1], (int, float)) else result[1],
                            "similarity": round(result[2], 2) if isinstance(result[2], (int, float)) else result[2],
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "result": "success" if "Identified and logged in!" in result[0] else "fail"
                        })
                        break  # Exit retry loop on success
                    except Exception as e:
                        print(f"Retrying {test_emotion} in iteration {iteration} due to error: {e}")
                        time.sleep(0.1)  # Short delay to avoid spamming retries

        # Clean up after each iteration
        self.delete_user(username)

        # Save to CSV
        output_path = os.path.join(os.path.dirname(__file__), "recognition_results.csv")
        with open(output_path, mode='w', newline='') as csv_file:
            fieldnames = ['iteration', 'train_emotion', 'test_emotion', 'match_score', 'similarity', 'timestamp', 'result']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"Total tests run: {len(results)}")
        print(f"CSV saved to: {output_path}")
        return output_path

        
        #self.add_user(data[0][1], data[1][1], data[2][1], data[3][0])
        #identified = self.recognise(data[0][1], data[3][0])
        #identified = self.recognise(data[1][1], data[3][0])
        #identified = self.recognise(data[2][1], data[3][0])
        #identified = self.recognise(data[3][1], data[3][0])
        #identified = self.recognise(data[4][1], data[3][0])
        #self.delete_user(data[3][0])
        #return data
        
    #def test(self):
        # Load 100 wav samples from a directory
        wav_dir = os.path.join(os.path.dirname(__file__), "test_wav", "china")
        file_names = sorted(os.listdir(wav_dir))  # Ensure consistent ordering

        assert len(file_names) >= 100, "Expected at least 100 .wav files"

        enrolled_users = {}
        results = []

        # Step 1: Enroll first 100 users
        for i in range(100):
            file_name = file_names[i]                        
            username = os.path.splitext(file_name)[0]       
            file_path = os.path.join(wav_dir, file_name)

            with open(file_path, 'rb') as f:
                wav_data = f.read()

            # Enroll using slightly modified copies to simulate three unique samples
            self.add_user(
                wav_data,
                self.mutate_bytes(wav_data, 1000),
                self.mutate_bytes(wav_data, 2000),
                username
            )

            enrolled_users[username] = wav_data

        # Step 2: Test recognition for each file against each enrolled user
        iteration = 0  # Initialize iteration counter
        for test_index in range(100):
            iteration += 1
            test_filename = file_names[test_index]
            test_user = os.path.splitext(test_filename)[0] 

            for compare_index in range(100):  # Changed from 30 to 100 for exactly 10,000 results
                compare_filename = file_names[compare_index]        
                compare_user = os.path.splitext(compare_filename)[0]  
                is_true_user = (test_user == compare_user)
                is_enrolled = compare_index < 100  # Changed from 30 to 100
                with open(os.path.join(wav_dir, compare_filename), 'rb') as f:
                    test_data = f.read()

                while True:
                    try:
                        result = self.recognise(test_data, test_user)
                        print(f"Iteration {iteration}: Result:", result)
                        results.append({
                            "iteration": iteration,
                            "test_file_user": test_user,
                            "compare_file_user": compare_user,
                            "is_true_user": is_true_user,
                            "is_enrolled": is_enrolled,
                            "match_score": round(result[1], 2) if isinstance(result[1], (int, float)) else result[1],
                            "similarity": round(result[2], 2) if isinstance(result[2], (int, float)) else result[2],
                            "result": "success" if "Identified and logged in!" in result[0] else "fail",
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        })
                        break  # Exit loop on success
                    except Exception as e:
                        print(f"Iteration {iteration}: Retrying: {test_filename} vs {compare_filename} due to error: {e}")
                        time.sleep(0.1)

        # Save results to CSV
        output_path = os.path.join(os.path.dirname(__file__), "multi_user_results.csv")
        with open(output_path, mode='w', newline='') as csv_file:
            fieldnames = ['iteration', 'test_file_user', 'compare_file_user', 'is_true_user', 'is_enrolled','match_score', 'similarity', 'result', 'timestamp']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"Total tests run: {len(results)}")
        print(f"CSV saved to: {output_path}")

        #Delete all enrolled users (100 users)
        print("Cleaning up enrolled users...")
        cleanup_errors = []
        for i in range(100):  # Changed from 5 to 100 to match enrolled users
            username = os.path.splitext(file_names[i])[0]  # e.g., "user_0"
            try:
                self.delete_user(username)
                print(f"Successfully deleted user: {username}")
            except Exception as e:
                error_msg = f"Error deleting user {username}: {e}"
                print(error_msg)
                cleanup_errors.append(error_msg)

        if cleanup_errors:
            print(f"Cleanup completed with {len(cleanup_errors)} errors")
        else:
            print("Cleanup completed successfully")

        return output_path    

    def recognise(self, voice, username):
        
        confidence = "N/A"
        similarity = 0.0  # Initialize similarity variable
        
        # Voice Authentication
        VOICENAMES = [ name for name in os.listdir(os.path.dirname(__file__) + self.VOICEPATH) if os.path.isdir(os.path.join(os.path.dirname(__file__) + self.VOICEPATH, name)) ]

        #IPFS Read Biometric Database
        enc_voice_dict = requests.get("https://ipfs.io/ipfs/" + self.IPFS_HASH).text
        voice_dict = self.fernet.decrypt(bytes(enc_voice_dict, encoding='utf-8'))
        with open(os.path.dirname(__file__) + "\\gmm_models\\voice_auth.gmm", 'wb') as file:
            file.write(voice_dict)

        if username in VOICENAMES:
            userIndex = VOICENAMES.index(username)
            arr = [VOICENAMES[userIndex]]
            try:
                # load model 
                model = pickle.load(open(os.path.dirname(__file__) + self.MODEL,'rb'))

                # reading audio files of speaker
                (sr, audio) = read(io.BytesIO(voice))
                    
                # extract 40 dimensional MFCC
                vector = extract_features(audio,sr).flatten()
                vector = self.pad_or_trim(vector, target_length=6000)
                test_audio = vector.reshape(1, -1)

                # Calculate similarity by comparing with stored user voice features
                try:
                    # Get stored voice features for the user
                    user_voice_features = []
                    if username in self.VOICEDICT:
                        for stored_voice in self.VOICEDICT[username]:
                            (stored_sr, stored_audio) = read(io.BytesIO(stored_voice))
                            stored_vector = extract_features(stored_audio, stored_sr).flatten()
                            stored_vector = self.pad_or_trim(stored_vector, target_length=6000)
                            user_voice_features.append(stored_vector)
                    
                    if user_voice_features:
                        # Calculate cosine similarity between test audio and stored voice features
                        from sklearn.metrics.pairwise import cosine_similarity
                        
                        similarities = []
                        for stored_feature in user_voice_features:
                            # Reshape for cosine similarity calculation
                            test_reshaped = test_audio.reshape(1, -1)
                            stored_reshaped = stored_feature.reshape(1, -1)
                            
                            # Calculate cosine similarity (returns value between 0 and 1)
                            cos_sim = cosine_similarity(test_reshaped, stored_reshaped)[0][0]
                            similarities.append(cos_sim)
                        
                        # Take the maximum similarity as the final similarity score
                        similarity = max(similarities) if similarities else 0.0
                        
                except Exception as e:
                    print("Similarity calculation error:", e)
                    similarity = 0.0

                # Predict class index
                pred = model.predict(test_audio)

                # decode predictions
                le = preprocessing.LabelEncoder()   
                le.fit(arr)
                try:
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(test_audio)[0]
                        user_index = le.transform([username])[0]
                        confidence = proba[user_index]
                except Exception as e:
                    print("Confidence error:", e)
                    confidence = "N/A"

                print("confidence", confidence)
                print("similarity", similarity)
                identity = le.inverse_transform(pred)[0]


                # if voice not recognized than terminate the process
                if identity == 'unknown':
                    print("Not Recognised!")
                    return ("Not Found", confidence, similarity)
                else:
                    print( "Recognized as - ", username)
                    return ("Identified and logged in!", confidence, similarity)
                    
            except Exception as e:
                print("Stopped", e)
                return ("Not Found", confidence, similarity)
        else:
            return ("Username Missing", confidence, similarity)

    def delete_user(self, username):

        try:
            name = username 

            users = [ name for name in os.listdir(os.path.dirname(__file__) + self.VOICEPATH) if os.path.isdir(os.path.join(os.path.dirname(__file__) + self.VOICEPATH, name)) ]
            
            if name not in users or name == "unknown":
                print('No such user !!')
                return "No user"

            [os.remove(path) for path in glob.glob(os.path.dirname(__file__) + self.VOICEPATH + name + '/*')]
            os.removedirs(os.path.dirname(__file__) + self.VOICEPATH + name)

            if os.path.isfile(os.path.dirname(__file__) + "\\gmm_models\\voice_auth.gmm"): 
                os.remove(os.path.dirname(__file__) + "\\gmm_models\\voice_auth.gmm")
                
            voice_dir = [ name for name in os.listdir(os.path.dirname(__file__) + self.VOICEPATH) if os.path.isdir(os.path.join(os.path.dirname(__file__) + self.VOICEPATH, name)) ]
            X = []
            Y = []
            for voice in self.VOICEDICT[username]:
                # reading audio files of speaker
                (sr, audio) = read(io.BytesIO(voice))
                            
                # extract 40 dimensional MFCC
                vector = extract_features(audio,sr)
                vector = vector.flatten()
                X.append(vector)
                Y.append(name)
                
            X = np.array(X, dtype=object)

            le = preprocessing.LabelEncoder()
            le.fit(Y)
            Y_trans = le.transform(Y)
            clf = LogisticRegression(random_state=0).fit(X.tolist(), Y_trans)

            if os.path.isfile(os.path.dirname(__file__) + "\\gmm_models\\voice_auth.gmm"): 
                os.remove(os.path.dirname(__file__) + "\\gmm_models\\voice_auth.gmm")
            # saving model
            pickle.dump(clf, open(os.path.dirname(__file__) + '\\gmm_models\\voice_auth.gmm', 'wb'))

            print('User ' + name + ' deleted successfully')
            return "deleted"

        except Exception as e:
            print("Error encountered:", e)
            return "deleted"



