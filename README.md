# Syborg

An AI-powered bot that detects Sybil bots and anomalies in blockchain networks.

Syborg leverages machine learning techniques to analyze behavioral data across a blockchain network. It classifies wallets into three categories:

- 🟢 Normal: Legitimate, healthy wallet behavior
- 🟡 Anomalous: Unusual behaviour but not a Sybil Bot
- 🔴 Sybil: Potentially malicious wallets exhibiting bot-like behavior

Using features such as transaction frequency, timing patterns, and wallet interaction graphs, Syborg aims to bring intelligent anomaly detection to Web3.

## How It Works

- Users upload wallet data (e.g., number of sent transactions) via a simple HTML interface.
- The data is sent to the backend for processing and analysis.
- Syborg uses two machine learning models:
    - Random Forest for detecting Sybil bots.
    - Isolation Forest for identifying anomalies.
- These models evaluate behavioral features such as:
    - Transaction frequency
    - Timing patterns
    - Wallet interconnections
- Syborg classifies each wallet as:
    - Normal
    - Anomalous
    - Sybil bot
- The classification results are returned to the frontend and displayed to the user.

## Future Developments

- Real-time integration with blockchain nodes
- Web dashboard for visualizing detected anomalies
- On-chain alert triggers

## Team

Developed by VRM-Tech:

- Valliammai Palaniappan
- Riya Jain
- Mathushawlika Muthukumar
