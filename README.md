# Voice-dApp: Decentralised Voice Biometric Authentication

**One-to-many voice biometric authentication, backed by decentralised storage and smart contract access controls.**

---

## 🚀 Overview

**Voice-dApp** is a web-based decentralised application (dApp) built with Flask that enables **voice biometric authentication** in a one-to-many setting. It combines classical voice feature modelling with blockchain-backed access control and distributed storage to create a system that is:

- **Secure & tamper-evident** - biometric models are encrypted and pinned to IPFS  
- **Decentralised** - smart contracts on Ethereum manage client permissions  
- **Transparent & auditable** - able to verify storage integrity and access history  

### Key Features

1. **Voice Enrolment & Recognition**  
   - Users enrol by uploading short voice samples  
   - MFCC (Mel-Frequency Cepstral Coefficients) features are extracted  
   - A scikit-learn model (e.g. SVM, GMM, etc.) is trained to discriminate speakers  

2. **Secure Storage via IPFS + Encryption**  
   - The trained biometric model is encrypted using **Fernet** (symmetric encryption)  
   - The encrypted file is pinned to IPFS via Pinata, producing a content-addressable CID  
   - This ensures tamper-evidence: any change to the content changes the hash  

3. **Access Control via Ethereum Smart Contract**  
   - A smart contract (compiled with `py-solc-x`) maintains authorised client identities  
   - Interaction with the contract is done via **Web3.py**  
   - Only authorised clients can invoke recognition or enrolment routes  

4. **Utilities & Analyses**  
   - Audio I/O handled with **PyAudio** / **OpenCV (if using visual/audio features)**  
   - Signal processing via **NumPy / SciPy**  
   - Optional analysis scripts to generate ROC / DET / threshold performance plots using **Matplotlib / Seaborn**  

---

## 🏗 Architecture

Below is a high-level view of how the components interact:

```
[Client App / Web UI]
       │
       ▼
[Flask Backend (API Endpoints)]
   ├── Enrolment → extract MFCC → update model  
   ├── Recognition → feature extraction + model inference  
   └── Helper routes (e.g. retrieving IPFS CID, status)
       │
       ▼
[Encryption / IPFS Module]
   ├── Encrypt model (Fernet)  
   └── Pin to IPFS via Pinata → store CID  

[Smart Contract Module]
   ├── Deploy / compile smart contract  
   └── Web3 interface for checking permissions  
       ↕  
[Ethereum Blockchain (e.g. testnet or local chain)]
```

---

## 🧩 Prerequisites & Setup

### Requirements

- Python 3.8+  
- A running Ethereum node or testnet (e.g. Ganache, Infura endpoint)  
- Pinata account (for IPFS pinning)  
- Audio interface or microphone (for sample recording)  

### Installation Steps

1. **Clone the repo**  
   ```bash
   git clone https://github.com/norbertdajnowski/voice-dapp.git
   cd voice-dapp
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables / settings**

   Example (in `.env` or config file):

   ```
   PINATA_API_KEY=your_pinata_key
   PINATA_SECRET=your_pinata_secret
   ETH_NODE_URI=https://<your_node_url>
   CONTRACT_ADDRESS=<deployed_contract_address>
   FERNNET_KEY=<your_secret_key>
   ```

4. **Compile / deploy smart contract**  
   Use the provided compilation scripts (via `py-solc-x`) and deploy to your Ethereum network.

5. **Run server**  
   ```bash
   python runserver.py
   ```

6. **Access the UI / API**  
   Visit `http://localhost:5000` (or whichever host:port you set)

---

## 📊 Usage Examples

- **Enrol a new speaker**  
  Send a short voice sample (e.g. WAV, ~2–5 seconds) to the `/enrol` route. The system extracts features, updates the model, encrypts it, pins to IPFS, and registers the new CID.

- **Authenticate a user**  
  Send a voice sample to `/recognise` (with client credentials). The backend retrieves the encrypted model (from IPFS via CID), decrypts it, runs inference, and returns a match score or decision.

- **Check permission**  
  The backend first queries the smart contract to confirm if the requesting client is authorised before performing recognition.

- **Performance analyses**  
  Use the scripts under the `analysis/` folder to generate ROC curves, DET plots, and evaluate thresholds based on a test dataset.

---

## 🛠️ Directory Structure

```
├── runserver.py              - Entry point for launching the Flask app  
├── requirements.txt  
├── compiled_code.json        - Compiled smart contract artefacts  
├── encrypted_voice_auth.gmm  - Sample encrypted biometric model (for demo)  
├── project/                   - Core modules (API, crypto, IPFS, contract)  
├── analysis/                  - Scripts for evaluation / plotting  
└── README.md                  - This file  
```

---

## ✅ Why This Matters

- **Hybrid security model**: Combines biometric authentication (something you *are*) with blockchain-based access control  
- **Decentralised auditability**: By pinning models to IPFS, tampering or modification is evident  
- **Extensibility**: You may adapt the voice model to other modalities (e.g. face, keystroke) or swap in neural networks or more advanced pipelines  
- **Research utility**: Useful as a proof-of-concept or a baseline system in biometric + blockchain research  

---

## 🚧 Limitations & Future Work

- Current model is a classical ML approach (SVM, GMM) - may not scale well to large populations  
- No real-time streaming support - batch sample only  
- Key management is simplistic (single symmetric key) - could be improved with hierarchical key systems or identity-based encryption  
- Smart contract lacks rich governance features (revocation, multi-sig)  
- UI is minimal; integration with a polished frontend / mobile app is missing  

---

## 🤝 Contributing

You are welcome to contribute! Possible areas:

- Add support for deep learning models (e.g. PyTorch / TensorFlow)  
- Real-time / streaming voice recognition  
- Improved UI / frontend (React, Vue, mobile support)  
- Enhanced smart contract features (revocation, role-based access)  
- Better key management (asymmetric, threshold cryptography)  
- More thorough testing, CI/CD, containerisation (Docker)  

To contribute:

1. Fork the repository  
2. Create your feature branch (`git checkout -b feature/xyz`)  
3. Commit your changes & write tests  
4. Submit a pull request and describe your changes  

---

## 📚 References & Inspirations

- MFCC & speaker recognition techniques  
- IPFS and decentralised storage architecture  
- Ethereum / smart contracts + Web3 access control  
- Cryptographic storage (Fernet, symmetric encryption)  

---

## 📄 Licence & Acknowledgements

Please ensure you include a licence file (e.g. MIT, Apache 2.0) to clarify reuse permissions. If you used or adapted existing code, add attributions accordingly.  
