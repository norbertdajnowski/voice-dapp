# -*- coding: utf-8 -*-
from flask import flash
import pyaudio
import wave
import cv2
import os
import glob
from cryptography.fernet import Fernet
import pickle
import time
import io
import requests

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
        
        # Test Pinata connection on initialization
        self._test_pinata_connection()

    def _test_pinata_connection(self):
        """Test Pinata API connection and credentials"""
        try:
            # Test with a simple API call
            test_response = pinata.test_authentication()
            print("Pinata connection test:", test_response)
            if 'authenticated' in test_response and test_response['authenticated']:
                print("✅ Pinata API connection successful")
            else:
                print("❌ Pinata API authentication failed")
        except Exception as e:
            print(f"❌ Pinata API connection error: {e}")
            print("Please check your Pinata API credentials")

    def add_user(self, voice1, voice2, voice3, username, additional_voices=None):   

        result = False
        source = self.VOICEPATH + username
        absolute_path = os.path.dirname(__file__) + self.VOICEPATH + username
        os.mkdir(absolute_path)
        
        # Combine base voices with any additional training samples
        voice_dir = [voice1, voice2, voice3]
        if additional_voices:
            voice_dir.extend(additional_voices)
        
        self.VOICEDICT[username] = voice_dir
        X = []
        Y = []
        for name in voice_dir:
            # reading audio files of speaker
            (sr, audio) = read(io.BytesIO(name))
                    
            # extract 40 dimensional MFCC
            vector = extract_features(audio,sr)
            vector = vector.flatten()
            X.append(vector)
            Y.append(name)
        X = np.array(X, dtype=object)

        le = preprocessing.LabelEncoder()
        le.fit(Y)
        Y_trans = le.transform(Y)
        
        # Use Random Forest for better voice recognition performance
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        ).fit(X.tolist(), Y_trans)

        if os.path.isfile(os.path.dirname(__file__) + "\\gmm_models\\voice_auth.gmm"): 
            os.remove(os.path.dirname(__file__) + "\\gmm_models\\voice_auth.gmm")
            if self.IPFS_HASH != "":
                response = pinata.remove_pin_from_ipfs(self.IPFS_HASH)   
                print(response)  
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
            try:
                response = pinata.pin_file_to_ipfs(os.path.dirname(__file__) + '\\gmm_models\\encrypted_voice_auth.gmm')   
                print("Pinata API Response:", response)  # Debug output
                
                # Check if response contains the expected key
                if 'IpfsHash' in response:
                    self.IPFS_HASH = response['IpfsHash']
                    print(f"IPFS Hash: {self.IPFS_HASH}")
                elif 'ipfsHash' in response:  # Check for lowercase version
                    self.IPFS_HASH = response['ipfsHash']
                    print(f"IPFS Hash: {self.IPFS_HASH}")
                else:
                    print("Error: IPFS Hash not found in response")
                    print("Available keys:", list(response.keys()) if isinstance(response, dict) else "Response is not a dictionary")
                    return False

                print(username + ' added successfully') 
                result = True
            except Exception as e:
                print(f"Error uploading to IPFS: {e}")
                print(f"Response type: {type(response)}")
                print(f"Response content: {response}")
                return False
        
        features = np.asarray(())
        return result

    def recognise(self, voice, username):
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
                vector = extract_features(audio,sr)
                vector = vector.flatten()
                test_audio = vector

                # predict with model
                pred = model.predict(test_audio.reshape(1,-1))

                # decode predictions
                le = preprocessing.LabelEncoder()
                le.fit(arr)
                identity = le.inverse_transform(pred)[0]

                # Get match probabilities
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(test_audio.reshape(1,-1))[0]
                    user_index = le.transform([username])[0]
                    confidence = probabilities[user_index]
                    
                    # Set confidence threshold for authentication (0.7-0.9 range)
                    confidence_threshold = 0.7
                    
                    # Check if confidence meets threshold
                    if confidence >= confidence_threshold:
                        print(f"Recognized as - {username} with confidence: {confidence:.3f}")
                        return ("Identified and logged in!", confidence)
                    else:
                        print(f"Low confidence match: {confidence:.3f} (threshold: {confidence_threshold})")
                        return ("Not Found", confidence)
                else:
                    confidence = 0.0
                    print("Model does not support probability prediction")
                    return ("Not Found", confidence)
                    
            except Exception as e:
                print("Stopped", e)
                return ("Not Found", confidence)
        else:
            return ("Username Missing", confidence)

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
                print("test")
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
            
            # Use Random Forest for better voice recognition performance
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            ).fit(X.tolist(), Y_trans)

            if os.path.isfile(os.path.dirname(__file__) + "\\gmm_models\\voice_auth.gmm"): 
                os.remove(os.path.dirname(__file__) + "\\gmm_models\\voice_auth.gmm")
            # saving model
            pickle.dump(clf, open(os.path.dirname(__file__) + '\\gmm_models\\voice_auth.gmm', 'wb'))

            print('User ' + name + ' deleted successfully')
            return "deleted"

        except:
            print("Error encountered")
            return "deleted"

    def retrain_user(self, username, additional_voices):
        """
        Retrain the model with additional voice samples for an existing user
        This helps improve recognition accuracy
        """
        try:
            if username not in self.VOICEDICT:
                return "User not found"
            
            # Add new voice samples to existing ones
            self.VOICEDICT[username].extend(additional_voices)
            
            # Get all users and their voice samples
            VOICENAMES = [name for name in os.listdir(os.path.dirname(__file__) + self.VOICEPATH) 
                          if os.path.isdir(os.path.join(os.path.dirname(__file__) + self.VOICEPATH, name))]
            
            X = []
            Y = []
            
            for user in VOICENAMES:
                if user in self.VOICEDICT:
                    for voice_data in self.VOICEDICT[user]:
                        (sr, audio) = read(io.BytesIO(voice_data))
                        vector = extract_features(audio, sr)
                        vector = vector.flatten()
                        X.append(vector)
                        Y.append(user)
            
            X = np.array(X, dtype=object)
            
            le = preprocessing.LabelEncoder()
            le.fit(Y)
            Y_trans = le.transform(Y)
            
            # Use Random Forest for better voice recognition performance
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            ).fit(X.tolist(), Y_trans)
            
            # Save retrained model
            if os.path.isfile(os.path.dirname(__file__) + "\\gmm_models\\voice_auth.gmm"): 
                os.remove(os.path.dirname(__file__) + "\\gmm_models\\voice_auth.gmm")
                if self.IPFS_HASH != "":
                    response = pinata.remove_pin_from_ipfs(self.IPFS_HASH)   
                    print(response)  
            
            pickle.dump(clf, open(os.path.dirname(__file__) + '\\gmm_models\\voice_auth.gmm', 'wb'))
            
            # Encrypt and upload to IPFS
            try:
                with open(os.path.dirname(__file__) + "\\gmm_models\\voice_auth.gmm", "rb") as voice_dict:
                    safe_voice_dict = open(os.path.dirname(__file__) + "\\gmm_models\\encrypted_voice_auth.gmm", "w")
                    encrypted_voice = self.fernet.encrypt(voice_dict.read())
                    safe_voice_dict.write(str(encrypted_voice, 'utf-8'))
                    safe_voice_dict.close()
            except Exception as e:
                print("Error during encryption:", e)
                
            if os.path.isfile(os.path.dirname(__file__) + "\\gmm_models\\encrypted_voice_auth.gmm"):
                try:
                    response = pinata.pin_file_to_ipfs(os.path.dirname(__file__) + '\\gmm_models\\encrypted_voice_auth.gmm')   
                    print("Pinata API Response:", response)  # Debug output
                    
                    # Check if response contains the expected key
                    if 'IpfsHash' in response:
                        self.IPFS_HASH = response['IpfsHash']
                        print(f"IPFS Hash: {self.IPFS_HASH}")
                    elif 'ipfsHash' in response:  # Check for lowercase version
                        self.IPFS_HASH = response['ipfsHash']
                        print(f"IPFS Hash: {self.IPFS_HASH}")
                    else:
                        print("Error: IPFS Hash not found in response")
                        print("Available keys:", list(response.keys()) if isinstance(response, dict) else "Response is not a dictionary")
                        return "error"
                except Exception as e:
                    print(f"Error uploading to IPFS: {e}")
                    print(f"Response type: {type(response)}")
                    print(f"Response content: {response}")
                    return "error"
            
            print(f'User {username} retrained successfully with {len(additional_voices)} additional samples')
            return "retrained"
            
        except Exception as e:
            print(f"Error during retraining: {e}")
            return "error"



