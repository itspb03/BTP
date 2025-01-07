# BTP

# Analysis and Prediction of Strata Monitored Data

## Overview
This repository contains the implementation of my Bachelor Thesis Project: **Analysis and Prediction of Strata Monitored Data using Machine Learning and Deep Learning Models**. The project focuses on visualizing and analyzing real-time data from strata monitoring instruments and applying predictive models to aid in the development of safety measures in a chromite mine located in the Baula Nausahi ultramafic complex (BNUC), Odisha.

---

## Key Features

1. **Data Visualization Dashboard**
   - Developed an interactive dashboard using **Dash**, **NumPy**, and **Pandas**.
   - Supports extraction and plotting of real-time data from:
     - **10 Pressure Cells**
     - **7 MPBX (Multiple Point Borehole Extensometers)**
   - Enables monitoring of key metrics like deformation and pressure trends.

2. **Predictive Modeling**
   - Applied machine learning and deep learning models to forecast pressure data:
     - **LSTM (Long Short-Term Memory)**
     - **GRU (Gated Recurrent Unit)**
     - **Random Forest (RF)**
   - Achieved best results with GRU:
     - **MSE (Mean Squared Error): 0.0049**
     - **R² (Coefficient of Determination): 0.80**
   - Model comparison:
     - **LSTM**: MSE: 0.0124, R²: 0.48
     - **Random Forest**: MSE: 0.0276, R²: 0.45

3. **Application Goals**
   - Interpreting strata behavior trends.
   - Enhancing safety measures by providing predictive insights.
   - Supporting real-time decision-making with a scalable framework.

---

## Project Structure

```
.
├── data/               # Input datasets from Pressure Cells and MPBX
├── models/             # Machine Learning and Deep Learning models
├── dashboard/          # Dash application files
├── results/            # Model evaluation metrics and plots
├── README.md           # Project overview and details (this file)
└── requirements.txt    # Python dependencies
```

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://https://github.com/itspb03/BTP.git
   ```
2. Navigate to the project directory:
   ```bash
   cd BTP
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. **Run the Dashboard**:
    *(Make sure to update file path in app.py)*
   ```bash
   python dashboard/app.py
   ```
   Open the app in your browser at `http://localhost:8050`.

3. **Train Models**:
   - Scripts for training LSTM, GRU, and RF models are in the `models/` directory.
   - Example:
     ```bash
     python models/LSTM.py && python models/GRU.py && python models/RF.py
 
     ```



---

## Results
- Best-performing model: **GRU**
  - **MSE**: 0.0049
  - **R²**: 0.80
- Dashboard provides a user-friendly interface for real-time monitoring of deformation and pressure trends.

---

## Future Work
- Extend predictive modeling to MPBX data.
- Integrate additional monitoring instruments.
- Enhance the dashboard with more analytics features.

---

## License
This project is licensed under the MIT License. See `LICENSE` for details.

---


