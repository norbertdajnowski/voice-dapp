# -*- coding: utf-8 -*-
from project import app
from flask import render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import DataRequired
from cryptography.fernet import Fernet

from project.models.voice import voice

from project.models.deploy_contract import web3Connect

web3Interface = web3Connect()
contract = web3Interface.deployContract()

voiceObj = voice()

class CreateForm(FlaskForm):
    text = StringField('name', validators=[DataRequired()])


@app.route('/')
def start():
    return render_template('printer/index.html')

@app.route('/about')
def about():
    return render_template('printer/about.html')

@app.route('/login')
def login():
    return render_template('printer/login.html')

@app.route('/register')
def register():
    return render_template('printer/register.html')

@app.route('/delete')
def delete():
    return render_template('printer/delete.html')

@app.route('/addVoice', methods=['GET', 'POST'])
def addVoice():
    voice1 = request.files['voice1'].read()
    voice2 = request.files['voice2'].read()
    voice3 = request.files['voice3'].read()
    username = request.files['username'].read().decode("utf-8") 

    if (voiceObj.add_user(voice1, voice2, voice3, username) == True):
        tx_hash = contract.functions.add(str(hash(web3Interface.clientAddress))).build_transaction({
            "chainId": web3Interface.chain_id,
            "gasPrice": web3Interface.web3.eth.gas_price,
            "from": web3Interface.clientAddress,
            "nonce": web3Interface.web3.eth.get_transaction_count(web3Interface.clientAddress)
        })

        web3Interface.nonce += 1

        signed_tx = web3Interface.web3.eth.account.sign_transaction(tx_hash, private_key=web3Interface.private_key)
        
        send_tx = web3Interface.web3.eth.send_raw_transaction(signed_tx.raw_transaction)

        tx_receipt = web3Interface.web3.eth.wait_for_transaction_receipt(send_tx)

        print(tx_receipt)

    return render_template('printer/index.html')


@app.route('/recognise', methods=['GET', 'POST'])
def recognise():
    if (contract.functions.check(str(hash(web3Interface.clientAddress))).call() == True):
        voice = request.files['voice1'].read()
        username = request.files['username'].read().decode("utf-8") 
        loginResult = voiceObj.recognise(voice, username)
        print("login result -", loginResult)
        if loginResult == "Not Found" or loginResult == None:
            return f'We could not identify your account, refresh and try again.'
        elif loginResult == "Username Missing":
            return f'Your username does not exist in the database'
        else:
            return f'Identification completed succesfully!'
    else:
        return f'Blockchain client not part of the biometric network'
    
@app.route('/test', methods=['GET', 'POST'])
def test():
    data = voiceObj.test()
    return "voiceObj.test()"

@app.route('/deleteVoice', methods=['GET', 'POST'])
def deleteVoice():
    username = request.files['username'].read().decode("utf-8") 
    deleteResult = voiceObj.delete_user(username)
    print("Delete user result - " + deleteResult)
    if deleteResult == "deleted":
        tx_hash = contract.functions.remove(str(hash(web3Interface.clientAddress))).build_transaction({
            "chainId": web3Interface.chain_id,
            "gasPrice": web3Interface.web3.eth.gas_price,
            "from": web3Interface.clientAddress,
            "nonce": web3Interface.nonce
        })

        web3Interface.nonce += 1

        signed_tx = web3Interface.web3.eth.account.sign_transaction(tx_hash, private_key=web3Interface.private_key)
        
        send_tx = web3Interface.web3.eth.send_raw_transaction(signed_tx.raw_transaction)

        tx_receipt = web3Interface.web3.eth.wait_for_transaction_receipt(send_tx)

        print(tx_receipt)

        return f'User deleted succesfully'
    elif deleteResult == "No user":
        return f'Username was not found in our database'
    else:
        return f'Problem encountered during user deletion'
    
