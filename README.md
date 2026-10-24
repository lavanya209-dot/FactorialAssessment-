Does External Pressure Matter? Analysis code
Analysis code for the manuscript:
L. P. Ramalingam, B. Ramalingam, S. Rathnasamy, U. Mani, P. Kathirvelu. Does External Pressure Matter? A Cell-Blocked Factorial Assessment of Temperature, C-Rate, Pressure and Cell-to-Cell Variation in Lithium Iron Phosphate Cells. Submitted to Energy Storage (Wiley), 2026.
The scripts reanalyse a public dataset in which six LiFePO4/graphite pouch cells (GSP655060Fe, 1600 mAh) were tested under every combination of three temperatures (10, 25 and 40 °C), four external confinement forces (300, 400, 500 and 600 N) and four C-rates (0.5C, 1C, 1.5C and 2C). This gives 288 single-cycle tests.
No new experiments were carried out for this work. All data come from the dataset below.
Data
The data are not included in this repository. Download them from Mendeley Data (licence CC BY 4.0):
•	C. Yan, Li-ion battery charge/discharge test data under multi-application scenarios, Mendeley Data, V1, 2024. https://doi.org/10.17632/dszd9mmd5j.1
•	Data article: C. Yan, X. Wu, Y. Yuan, Y. Xie, J. Wang, G. Gao, Y. Fan, Experimental data simulating lithium battery charging and discharging tests under different external constraint pressure conditions, Data in Brief 55 (2024) 110616. https://doi.org/10.1016/j.dib.2024.110616
Please cite both if you use the data.
Unzip the dataset into one folder. The scripts read the test conditions from the folder and file names, so keep the original structure, for example:
libd/
  10d/
    300N/
      0.5C/
        Battery No.1.xls
        ...
        Battery No.6.xls
      1C/ ...
    400N/ ...
  25d/ ...
  40d/ ...
10d means 10 °C, 300N means 300 N, and Battery No.1 means cell 1. The loader stops with an error if any condition cannot be read, so no default values are used.
Requirements
•	Python 3.10 or newer
•	Packages:
pip install pandas numpy scipy statsmodels matplotlib scikit-learn openpyxl xlrd
The results in the paper were produced with statsmodels 0.15.0. xlrd is needed for the .xls files.
Scripts
Run the scripts in this order from the same folder. Each script has a --selftest option that checks its logic on synthetic data, without needing the real dataset.
Step	Script	What it does	Main outputs
1	Data_2026_1.py	Loads the raw tester files, classifies charge, rest and discharge steps, builds a continuous time axis and counts cycles per file	out/cycle_summary.csv
2	Data_2026_Step2.py	Calculates one row of outcomes per test: capacity, energy, mean discharge voltage, energy efficiency, discharge-to-charge capacity ratio and 10 s DC resistance	out_full/per_cell_outcomes.csv
3	Data_2026_Step3_Annova.py	Cell-blocked three-way ANOVA, variance shares, assumption checks, mixed-model and paired pressure tests, figures	out_full/anova/
4	Data_2026_Step4_ML.py	Condition-mean lookup, polynomial ridge, Gaussian process and gradient boosting models, validated by leaving one cell, one temperature or one C-rate out; noise floor, pressure ablation and permutation importance	out_full/ml/
5	Data_2026_RunOrder.py	Recovers the start time of every test from the raw time stamps and checks pressure against the test sequence	out_full/runorder/
6	Data_2026_Protocol.py	Reads the actual protocol from the raw files: step sequence, CV cut-off current, rest times, sampling interval and start voltage	out_full/protocol/
7	Data_2026_Step5_RepeatedMeasures.py	Repeated-measures ANOVA with the cells as units (Greenhouse-Geisser corrected), session-level pressure test, residual intraclass correlation and test-history effects	out_full/rm/
8	Data_2026_Figures.py	Recreates Figs. 1 to 5 and Table 3 of the paper	out_full/figures/
Data_2026_Step2.py, Data_2026_RunOrder.py and Data_2026_Protocol.py import the loader from Data_2026_1.py, so keep all scripts in the same folder.
How to run
Replace the data path with the folder where you unzipped the dataset.
python Data_2026_1.py --data-path "path/to/libd" --out-dir out
python Data_2026_Step2.py --data-path "path/to/libd" --out-dir out_full
python Data_2026_Step3_Annova.py --input out_full/per_cell_outcomes.csv --out-dir out_full/anova
python Data_2026_Step4_ML.py --input out_full/per_cell_outcomes.csv --out-dir out_full/ml
python Data_2026_RunOrder.py --data-path "path/to/libd" --outcomes out_full/per_cell_outcomes.csv --out-dir out_full/runorder
python Data_2026_Protocol.py --data-path "path/to/libd" --out-dir out_full/protocol
python Data_2026_Step5_RepeatedMeasures.py
python Data_2026_Figures.py
Steps 1, 2, 5 and 6 read all 288 Excel files and take a few minutes each. Steps 3, 4, 7 and 8 use the small output tables and run quickly.
To check that the code works before downloading the data:
python Data_2026_1.py --selftest
python Data_2026_Step2.py --selftest
python Data_2026_Step3_Annova.py --selftest
python Data_2026_Step4_ML.py --selftest
python Data_2026_RunOrder.py --selftest
python Data_2026_Protocol.py --selftest
python Data_2026_Step5_RepeatedMeasures.py --selftest
python Data_2026_Figures.py --selftest
Which script produces which result in the paper
Paper item	Script
Table 1 protocol values (cut-off current, rests, sampling interval)	Data_2026_Protocol.py
Test order and dates (Section 2.2)	Data_2026_RunOrder.py
Table 2 and the variance shares in Fig. 1	Data_2026_Step3_Annova.py
Repeated-measures and session-level tests, Table 4	Data_2026_Step5_RepeatedMeasures.py
Residual dependence and test history (Sections 3.1 and 3.5)	Data_2026_Step5_RepeatedMeasures.py
Machine learning results (Section 3.6)	Data_2026_Step4_ML.py
Figs. 1 to 5 and Table 3	Data_2026_Figures.py (uses the outputs of the steps above)
Notes on the data
•	In the original test campaign, the six cells were tested together, the temperatures were run in the order 25, 10 and 40 °C, and pressure was always raised from 300 to 600 N. Pressure is therefore confounded with test session. See the paper for details.
•	No reset step was used between tests, so each test starts from the state left by the previous one.
•	The discharge-to-charge capacity ratio is stored in the column coulombic_efficiency for historical reasons. It is not a true coulombic efficiency and can exceed 1 (see Section 3.5 of the paper).
•	dc_resistance_ohm is a 10 s DC resistance taken at the start of discharge.
Funding
This work was supported by SASTRA Deemed to be University through the TRR fund (SASTRA-TRR-SEEE-4-27012026).
Contact
Dr. Parkavi Kathirvelu, Department of Electrical & Electronics Engineering, SASTRA Deemed to be University, Thanjavur, Tamil Nadu 613 401, India. E-mail: to_parkavi@eee.sastra.edu
