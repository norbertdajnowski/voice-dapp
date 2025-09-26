# Voice-dApp: Decentralised Voice Biometric Authentication

**One-to-many voice biometric authentication, backed by decentralised storage and smart contract access controls.**

---

## Overview

**Voice-dApp** is a web-based decentralised application (dApp) built with Flask that enables **voice biometric authentication** in a one-to-many setting. It combines classical voice feature modelling with blockchain-backed access control and distributed storage to create a system that is:

---

## Setup

### Requirements

- Python 3.8+  
- A running Ethereum node or testnet (e.g. Ganache, Infura endpoint)  
- Pinata account (for IPFS pinning)  
- Audio interface or microphone (for sample recording)  

### Installation Steps

1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
   
2. **Run Ganache Server**  

3. **Configure environment variables / settings**

   ```
   PINATA_API_KEY=your_pinata_key (from ganache)
   PINATA_SECRET=your_pinata_secret
   ETH_NODE_URI=https://<your_node_url>
   CONTRACT_ADDRESS=<deployed_contract_address>
   FERNNET_KEY=<your_secret_key>
   ```

4. **Compile / deploy smart contract**  
   Use the provided compilation scripts (via `py-solc-x`) and deploy to Ethereum network.

5. **Run server**  
   ```bash
   python runserver.py
   ```

6. **Access the UI / API**  
   Visit `http://localhost:5000` (or whichever host:port you set)

## References

- MFCC & speaker recognition techniques  
- IPFS and decentralised storage architecture  
- Ethereum / smart contracts + Web3 access control  
- Cryptographic storage
