# equipment_predictor_app.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import streamlit as st
from io import StringIO

# -------------------------
# Load Data
# -------------------------
api_data_str = """
timestamp,equipment_id,temperature,vibration,pressure,status,anomaly_score,predicted_failure_in_days
25-08-2025 09:48,1008,82.6,1.36,3.89,OK,0.251,7
25-08-2025 09:43,1012,82.63,2.03,5.04,OK,0.916,26
25-08-2025 09:38,1012,70.34,1.97,6.77,OK,0.853,24
25-08-2025 09:33,1001,72.69,2.1,2.49,OK,0.58,12
25-08-2025 09:28,1001,74.68,1.99,1.82,OK,0.52,28
25-08-2025 09:23,1015,72.81,1.51,6.87,OK,0.931,20
25-08-2025 09:18,1017,76.64,2.73,2.8,OK,0.932,24
25-08-2025 09:13,1002,74.7,2.94,4.43,OK,0.969,14
25-08-2025 09:08,1017,58.26,2.27,5.91,Warning,0.125,22
25-08-2025 09:03,1009,72.92,2.3,4.91,OK,0.161,16
25-08-2025 08:58,1012,74.77,2.07,4.89,Warning,0.677,29
25-08-2025 08:53,1004,84.9,2.48,5.11,Critical,0.541,19
25-08-2025 08:48,1013,66.69,0.78,8.78,Critical,0.018,23
25-08-2025 08:43,1004,77.04,2,7.67,Critical,0.062,3
25-08-2025 08:38,1007,80.63,1.71,3.96,OK,0.888,11
25-08-2025 08:33,1014,73.52,2.07,6.58,OK,0.869,23
25-08-2025 08:28,1009,57.45,2.55,5.76,OK,0.357,21
25-08-2025 08:23,1001,85.1,2.01,4.97,Warning,0.891,20
25-08-2025 08:18,1001,82.47,1.82,3.64,Warning,0.594,19
25-08-2025 08:13,1004,77.29,1.3,5.68,Warning,0.806,8
25-08-2025 08:08,1001,75.4,1.95,4.54,OK,0.427,20
25-08-2025 08:03,1010,57.19,2.34,4.43,Warning,0.113,8
25-08-2025 07:58,1002,82.75,1.58,3.85,OK,0.434,26
25-08-2025 07:53,1015,89.86,1,6.5,OK,0.457,3
25-08-2025 07:48,1003,63.86,3.14,3.85,Warning,0.551,10
25-08-2025 07:43,1009,88.45,2.17,3.86,OK,0.106,13
25-08-2025 07:38,1017,86.31,1.89,6.17,Critical,0.991,9
25-08-2025 07:33,1013,63.57,2.7,4.78,OK,0.244,6
25-08-2025 07:28,1003,71.06,2.07,4.82,Warning,0.199,15
25-08-2025 07:23,1004,61.05,2.67,6.55,OK,0.527,16
25-08-2025 07:18,1004,74.35,2.62,6.97,Warning,0.537,5
25-08-2025 07:13,1011,79.48,1.36,1.81,Warning,0.44,21
25-08-2025 07:08,1007,61.8,2.23,4.81,Critical,0.772,13
25-08-2025 07:03,1011,78.86,2.86,2.88,OK,0.517,19
25-08-2025 06:58,1011,81.39,1.5,4.42,Critical,0.61,2
25-08-2025 06:53,1010,81.53,1.51,4.99,OK,0.724,6
25-08-2025 06:48,1005,79.55,1.59,4.47,OK,0.025,27
25-08-2025 06:43,1013,62.72,2.45,3.46,OK,0.571,1
25-08-2025 06:38,1012,73.79,2.14,3.39,Warning,0.884,15
25-08-2025 06:33,1010,78.52,1.03,6.09,OK,0.017,15
25-08-2025 06:28,1007,89.61,2.17,7.3,Warning,0.405,14
25-08-2025 06:23,1019,75.02,1.87,1.91,Warning,0.926,7
25-08-2025 06:18,1019,79.26,1.94,4.51,OK,0.353,5
25-08-2025 06:13,1019,65.68,1.58,6.1,Critical,0.459,20
25-08-2025 06:08,1012,63.45,2.17,4.95,OK,0.982,25
25-08-2025 06:03,1015,62.17,2.53,3.33,Warning,0.199,26
25-08-2025 05:58,1005,70.8,1.68,4.28,OK,0.065,16
25-08-2025 05:53,1014,95.62,1.16,5.45,OK,0.859,17
25-08-2025 05:48,1012,70.98,1.85,4.54,OK,0.868,11
25-08-2025 05:43,1011,78.65,2.38,4.93,OK,0.472,15
25-08-2025 05:38,1005,80.57,1.79,4.97,OK,0.514,20
25-08-2025 05:33,1016,64.08,1.25,4.21,OK,0.836,16
25-08-2025 05:28,1018,70.65,2.38,6.22,OK,0.739,3
25-08-2025 05:23,1012,81.24,2.71,4.28,OK,0.265,6
25-08-2025 05:18,1005,79.28,2.01,3.49,OK,0.017,21
25-08-2025 05:13,1016,55.06,1.2,8.34,Warning,0.48,23
25-08-2025 05:08,1012,80.94,1.84,4.88,OK,0.859,13
25-08-2025 05:03,1002,69.49,2.08,1.1,Warning,0.896,1
25-08-2025 04:58,1006,69.75,1.6,6.88,OK,0.956,29
25-08-2025 04:53,1004,79.05,2.29,6.88,OK,0.88,21
25-08-2025 04:48,1013,90.23,1.55,5.69,OK,0.58,14
25-08-2025 04:43,1010,73,2.01,5.95,OK,0.445,2
25-08-2025 04:38,1017,86.79,1.89,8.24,OK,0.775,2
25-08-2025 04:33,1014,60.59,2.31,3.56,OK,0.74,27
25-08-2025 04:28,1015,76.31,1.99,5.2,Warning,0.538,15
25-08-2025 04:23,1008,69.59,1.9,7.95,Warning,0.087,27
25-08-2025 04:18,1018,77.82,1.6,8.27,OK,0.926,18
25-08-2025 04:13,1003,72.04,2.02,4.45,Warning,0.456,9
25-08-2025 04:08,1017,61.59,1.87,8.02,OK,0.363,26
25-08-2025 04:03,1007,66.13,1.56,6.2,OK,0.63,3
25-08-2025 03:58,1013,58.57,2.08,7.24,OK,0.674,4
25-08-2025 03:53,1016,61.93,2.18,6.15,Warning,0.211,15
25-08-2025 03:48,1005,79.8,1.83,5.46,Warning,0.837,5
25-08-2025 03:43,1004,72.49,2.47,8.31,Warning,0.29,20
25-08-2025 03:38,1010,77.31,2.13,4.21,OK,0.187,20
25-08-2025 03:33,1013,93.16,1.36,7.85,OK,0.406,5
25-08-2025 03:28,1010,72.04,2.67,8.09,Warning,0.731,10
25-08-2025 03:23,1014,81.34,2.38,7.04,OK,0.965,16
25-08-2025 03:18,1012,75.28,2.5,4.89,OK,0.809,25
25-08-2025 03:13,1001,102.63,2.2,5.37,OK,0.94,12
25-08-2025 03:08,1007,44.82,0.83,3.09,OK,0.141,25
25-08-2025 03:03,1009,67.82,2.55,4.32,OK,0.071,14
25-08-2025 02:58,1008,81.54,2.03,5.93,OK,0.558,22
25-08-2025 02:53,1013,67.47,2.18,2.86,OK,0.29,11
25-08-2025 02:48,1014,80.66,2.27,4.55,OK,0.828,28
25-08-2025 02:43,1012,74.4,2.97,7.92,OK,0.672,19
23-08-2025 23:33,1007,69.54,1.7,2.35,OK,0.771,16
23-08-2025 23:28,1003,73.97,2.45,6.09,OK,0.11,9
23-08-2025 23:23,1019,80.41,2.11,4.14,OK,0.027,2
23-08-2025 23:18,1019,95.37,1.85,3.16,OK,0.274,22
23-08-2025 23:13,1009,72.78,1.39,5.6,OK,0.269,27
23-08-2025 23:08,1010,92.69,1.76,2.34,OK,0.819,29
23-08-2025 23:03,1014,55.41,1.45,5.51,OK,0.711,3
23-08-2025 22:58,1018,76.48,2.38,3.3,Warning,0.982,10
23-08-2025 22:53,1015,71.66,2.61,2.98,Warning,0.712,20
23-08-2025 22:48,1001,69.68,2.46,5.09,OK,0.418,2
23-08-2025 22:43,1006,81.58,3.24,4.74,Warning,0.699,27
23-08-2025 22:38,1015,72.85,1.77,7.19,OK,0.999,24
23-08-2025 22:33,1006,54.75,2.3,3.07,Warning,0.71,10
23-08-2025 22:28,1018,59.12,1.41,7.06,OK,0.762,8
23-08-2025 22:23,1001,82.22,1.9,3.29,Warning,0.938,17
23-08-2025 22:18,1003,58.63,1.26,6.38,OK,0.114,26
23-08-2025 22:13,1017,82.49,2.28,6.66,OK,0.705,22
23-08-2025 22:08,1016,76.25,1.82,5.48,OK,0.851,1
23-08-2025 22:03,1008,73.3,0.97,3.81,OK,0.954,6
23-08-2025 21:58,1017,74.19,1.66,2.72,OK,0.211,7
23-08-2025 21:53,1008,80.15,2.9,2.07,OK,0.504,15
23-08-2025 21:48,1017,63.23,2.21,7.78,Warning,0.681,18
23-08-2025 21:43,1009,75.42,2.38,4.55,OK,0.247,9
23-08-2025 21:38,1012,63.6,2.06,3.34,OK,0.6,8
23-08-2025 21:33,1019,76.64,1.91,3.29,OK,0.258,25
23-08-2025 21:28,1004,81.46,2.01,5.25,Critical,0.333,10
23-08-2025 21:23,1011,76.66,2.12,4.41,Warning,0.442,29
23-08-2025 21:18,1004,81.09,1.45,2.12,Warning,0.529,4
23-08-2025 21:13,1014,68,2.38,6.3,OK,0.449,20
23-08-2025 21:08,1014,74.38,2.56,5.5,OK,0.493,28
23-08-2025 21:03,1006,72.62,1.9,7.76,OK,0.299,8
23-08-2025 20:58,1017,71.97,1.74,6.71,OK,0.159,1
23-08-2025 20:53,1002,66.99,1.63,0.35,OK,0.047,8
23-08-2025 20:48,1008,70.68,2.88,7.13,OK,0.196,23
23-08-2025 20:43,1004,63.87,1.77,6.81,OK,0.582,12
23-08-2025 20:38,1010,58.06,2.33,3.9,OK,0.786,23
23-08-2025 20:33,1007,67.99,1.91,4.35,OK,0.169,25
23-08-2025 20:28,1004,81.26,1.88,5.62,Warning,0.109,15
23-08-2025 20:23,1016,77.34,1.62,3.48,OK,0.749,13
23-08-2025 20:18,1014,62.43,1.51,4.64,Warning,0.305,16
23-08-2025 20:13,1016,70.94,2.37,6.5,OK,0.951,26
23-08-2025 20:08,1009,81.92,1.5,5.78,OK,0.714,23
23-08-2025 20:03,1004,70.52,2.23,4,OK,0.327,18
23-08-2025 19:58,1006,60.82,2.05,6.53,Warning,0.29,1
23-08-2025 19:53,1001,97.58,2.38,5.45,OK,0.524,17
23-08-2025 19:48,1007,73.33,1.7,5.4,Warning,0.079,16
23-08-2025 19:43,1001,63.36,2.54,4.23,OK,0.118,23
23-08-2025 19:38,1019,76.71,1.93,4.99,OK,0.176,5
23-08-2025 19:33,1008,74.14,3.3,3.84,Warning,0.775,20
23-08-2025 19:28,1003,59.59,1.19,2.09,OK,0.514,21
23-08-2025 19:23,1001,75.93,3.21,4.58,Warning,0.011,10
23-08-2025 19:18,1001,84.84,1.94,4.82,OK,0.674,26
23-08-2025 19:13,1007,72.67,2.11,5.21,OK,0.424,6
23-08-2025 19:08,1013,54.4,1.7,6.82,Warning,0.511,12
23-08-2025 19:03,1001,78.42,1.22,4.02,Critical,0.214,10
23-08-2025 18:58,1016,81.03,1.74,4.45,OK,0.552,22
23-08-2025 18:53,1014,69.14,1.82,6.15,Warning,0.979,17
23-08-2025 18:48,1004,69.21,2.09,3.87,OK,0.073,2
23-08-2025 18:43,1012,97.58,2.56,5.23,Warning,0.284,19
23-08-2025 18:38,1012,68.42,2.1,4.39,OK,0.243,20
23-08-2025 18:33,1017,63.78,2.1,4.63,OK,0.249,6
23-08-2025 18:28,1015,75.25,2.19,6.14,OK,0.379,28
23-08-2025 18:23,1010,80.75,1.12,4.44,OK,0.818,28
23-08-2025 18:18,1003,80.32,1.17,6.19,OK,0.913,27
23-08-2025 18:13,1012,50.32,1.25,2.07,OK,0.202,13
23-08-2025 18:08,1001,68.39,1.81,2.86,OK,0.543,9
23-08-2025 18:03,1001,86.53,1.06,1.52,Critical,0.637,12
23-08-2025 17:58,1010,78.27,1.66,3.25,OK,0.522,18
23-08-2025 17:53,1018,62.64,1.32,7.79,OK,0.434,8
23-08-2025 17:48,1015,87.42,2.24,4.97,OK,0.192,25
23-08-2025 17:43,1002,84.65,2.22,3.92,Critical,0.488,18
23-08-2025 17:38,1017,72.03,1.85,6.17,OK,0.401,25
23-08-2025 17:33,1008,83.79,1.05,6.39,OK,0.796,23
23-08-2025 17:28,1009,95,1.88,3.17,OK,0.764,25
23-08-2025 17:23,1011,64.46,2.16,8.75,OK,0.134,22
23-08-2025 17:18,1006,80.55,2.41,5.41,OK,0.089,2
23-08-2025 17:13,1014,94.86,2.88,5.6,OK,0.908,8
23-08-2025 17:08,1001,81.53,2.56,5.75,OK,0.536,10
23-08-2025 17:03,1007,67.46,2.08,5.74,OK,0.858,9
23-08-2025 16:58,1009,77.39,1.86,4.07,OK,0.058,28
23-08-2025 16:53,1011,88.7,1.63,8.17,OK,0.527,24
23-08-2025 16:48,1016,64.84,2.68,4.57,OK,0.539,18
23-08-2025 16:43,1019,74.7,2.08,6.44,Critical,0.861,8
23-08-2025 16:38,1002,72.57,2.32,4.29,Warning,0.098,19
23-08-2025 16:33,1015,68.69,1.7,4.02,OK,0.24,3
23-08-2025 16:28,1003,77.42,1.38,7.85,OK,0.407,9
23-08-2025 16:23,1004,73.99,1.8,4.84,OK,0.866,13
23-08-2025 16:18,1007,81.43,1.85,4.95,OK,0.939,9
23-08-2025 16:13,1012,86.41,2.1,5.43,OK,0.993,25

"""
maintenance_str = """
log_id,date_reported,equipment_id,issue_reported,priority,status,assigned_to,expected_resolution_date,actual_resolution_date,resolution_summary,downtime_hours,brakes_replaced,tires_replaced,chain_replaced,gears_replaced,electronics_replaced
1,2024-10-29,1018,Leakage,Low,Open,Alice,2025-08-25,2025-08-26,Replaced faulty valve,1.09,Yes,No,No,No,No
2,2024-10-30,1019,Overheating,Low,In Progress,David,2025-08-26,2025-08-27,Routine inspection,0.28,No,No,Yes,No,No
3,2024-11-31,1006,Pressure drop,Medium,Closed,David,2025-08-27,2025-08-28,Lubricated moving parts,2.02,No,No,No,No,No
4,2024-11-01,1002,Overheating,Medium,In Progress,David,2025-08-28,2025-08-29,Replaced faulty valve,0.57,No,No,Yes,No,No
5,2024-11-02,1013,Pressure drop,Low,Closed,Bob,2025-08-29,2025-08-30,Lubricated moving parts,0.05,No,No,No,No,No
6,2024-11-03,1011,Pressure drop,Low,Closed,David,2025-08-30,2025-08-31,Adjusted pressure,0.28,No,No,No,No,No
7,2024-11-04,1001,Pressure drop,Medium,Closed,Bob,2025-08-31,2025-09-01,Replaced faulty valve,1.45,No,Yes,No,No,No
8,2024-11-05,1002,Overheating,Low,Closed,Bob,2025-09-01,2025-09-02,Adjusted pressure,0.07,No,No,No,No,No
9,2024-11-06,1017,Pressure drop,High,Open,Charlie,2025-09-02,2025-09-03,Replaced faulty valve,0.16,No,No,No,No,No
10,2024-11-07,1008,Leakage,Medium,In Progress,David,2025-09-03,2025-09-04,Replaced faulty valve,1.18,No,No,Yes,No,No
11,2024-11-08,1002,Routine check,Medium,Open,David,2025-09-04,2025-09-05,Routine inspection,2.23,No,No,No,Yes,No
12,2024-11-09,1006,Routine check,Medium,Closed,David,2025-09-05,2025-09-06,Adjusted pressure,1.0,No,No,No,No,No
13,2024-11-10,1005,Pressure drop,Low,Open,Bob,2025-09-06,2025-09-07,Adjusted pressure,0.25,No,No,No,No,No
14,2024-11-11,1016,Routine check,High,In Progress,Bob,2025-09-07,2025-09-08,Adjusted pressure,0.05,No,Yes,No,No,No
15,2024-11-12,1008,Routine check,Low,In Progress,David,2025-09-08,2025-09-09,Replaced faulty valve,0.95,Yes,No,No,No,No
16,2024-11-13,1019,Pressure drop,Medium,Closed,Bob,2025-09-09,2025-09-10,Adjusted pressure,0.05,No,No,No,No,No
17,2024-11-14,1016,Pressure drop,Medium,In Progress,Bob,2025-09-10,2025-09-11,Adjusted pressure,0.95,No,No,No,No,No
18,2024-11-15,1013,Leakage,Low,Closed,Alice,2025-09-11,2025-09-12,Routine inspection,0.26,No,No,No,No,No
19,2024-11-16,1007,Vibration abnormal,Medium,Open,Alice,2025-09-12,2025-09-13,Lubricated moving parts,2.35,No,No,No,No,No
20,2024-11-17,1016,Overheating,High,Open,David,2025-09-13,2025-09-14,Lubricated moving parts,0.64,Yes,No,No,No,No
21,2024-11-18,1016,Pressure drop,Medium,In Progress,David,2025-09-14,2025-09-15,Adjusted pressure,0.25,No,No,No,No,No
22,2024-11-19,1012,Pressure drop,High,Closed,Alice,2025-09-15,2025-09-16,Adjusted pressure,1.92,No,No,No,No,No
23,2024-11-20,1014,Routine check,High,Closed,Bob,2025-09-16,2025-09-17,Adjusted pressure,0.29,No,No,No,No,No
24,2024-11-21,1018,Leakage,Medium,Open,Bob,2025-09-17,2025-09-18,Adjusted pressure,0.37,No,No,Yes,No,No
25,2024-11-22,1018,Routine check,Medium,Open,Charlie,2025-09-18,2025-09-19,Lubricated moving parts,1.8,No,No,No,No,Yes
26,2024-11-23,1016,Pressure drop,Medium,Closed,David,2025-09-19,2025-09-20,Routine inspection,0.37,No,No,No,No,No
27,2024-11-24,1016,Pressure drop,Medium,Open,David,2025-09-20,2025-09-21,Replaced faulty valve,0.53,No,Yes,No,No,No
28,2024-11-25,1013,Routine check,Low,Open,David,2025-09-21,2025-09-22,Adjusted pressure,0.43,No,No,No,No,No
29,2024-11-26,1009,Pressure drop,High,Open,Charlie,2025-09-22,2025-09-23,Adjusted pressure,0.12,No,Yes,No,No,No
30,2024-11-27,1009,Vibration abnormal,Medium,Open,Charlie,2025-09-23,2025-09-24,Lubricated moving parts,0.04,No,No,No,Yes,No
31,2024-11-28,1013,Overheating,Medium,In Progress,Bob,2025-09-24,2025-09-25,Adjusted pressure,1.29,No,No,No,Yes,No
32,2024-11-29,1004,Overheating,Medium,Open,Alice,2025-09-25,2025-09-26,Routine inspection,1.04,No,No,No,No,No
33,2024-11-30,1009,Overheating,Medium,In Progress,Alice,2025-09-26,2025-09-27,Routine inspection,0.33,No,No,No,No,No
34,2024-12-01,1015,Vibration abnormal,High,Closed,Alice,2025-09-27,2025-09-28,Adjusted pressure,0.35,No,No,No,No,No
35,2024-12-02,1014,Pressure drop,Medium,In Progress,Bob,2025-09-28,2025-09-29,Lubricated moving parts,2.9,No,No,No,No,No
36,2024-12-03,1019,Routine check,Low,Closed,Alice,2025-09-29,2025-09-30,Adjusted pressure,4.4,No,No,No,No,Yes
37,2024-12-04,1012,Leakage,Low,Open,Charlie,2025-09-30,2025-10-01,Adjusted pressure,0.64,Yes,No,No,No,No
38,2024-12-05,1006,Pressure drop,High,Closed,Charlie,2025-10-01,2025-10-02,Lubricated moving parts,1.62,No,No,No,No,No
39,2024-12-06,1003,Vibration abnormal,Medium,In Progress,David,2025-10-02,2025-10-03,Routine inspection,1.33,No,No,No,No,No
40,2024-12-07,1016,Vibration abnormal,High,Closed,Alice,2025-10-03,2025-10-04,Replaced faulty valve,0.35,No,No,No,No,No
41,2024-12-08,1017,Overheating,Low,Open,Charlie,2025-10-04,2025-10-05,Adjusted pressure,0.06,No,No,No,No,No
42,2024-12-09,1016,Pressure drop,Low,Closed,David,2025-10-05,2025-10-06,Routine inspection,0.15,No,Yes,No,No,No
43,2024-12-10,1006,Overheating,Low,In Progress,Alice,2025-10-06,2025-10-07,Adjusted pressure,1.82,No,No,No,No,Yes
44,2024-12-11,1008,Leakage,Medium,In Progress,Alice,2025-10-07,2025-10-08,Replaced faulty valve,1.31,No,No,No,No,No
45,2024-12-12,1019,Overheating,Medium,Closed,Charlie,2025-10-08,2025-10-09,Lubricated moving parts,1.09,No,No,No,Yes,No
46,2024-12-13,1002,Leakage,Low,Closed,David,2025-10-09,2025-10-10,Routine inspection,0.14,No,Yes,No,No,No
47,2024-12-14,1014,Vibration abnormal,Medium,In Progress,David,2025-10-10,2025-10-11,Replaced faulty valve,1.02,No,No,No,No,No
48,2024-12-15,1012,Overheating,Medium,Closed,Charlie,2025-10-11,2025-10-12,Routine inspection,0.34,No,No,No,No,No
49,2024-12-16,1005,Overheating,Medium,Open,Bob,2025-10-12,2025-10-13,Replaced faulty valve,1.47,No,No,Yes,No,No
50,2024-12-17,1018,Routine check,Low,Closed,Charlie,2025-10-13,2025-10-14,Routine inspection,0.65,Yes,No,No,No,No
51,2024-12-18,1007,Vibration abnormal,High,In Progress,Alice,2025-10-14,2025-10-15,Adjusted pressure,2.25,No,No,No,Yes,No
52,2024-12-19,1019,Pressure drop,Medium,Open,David,2025-10-15,2025-10-16,Lubricated moving parts,0.38,No,Yes,No,No,No
53,2024-12-20,1002,Leakage,Low,Closed,Bob,2025-10-16,2025-10-17,Replaced faulty valve,0.92,No,No,No,No,Yes
54,2024-12-21,1014,Overheating,High,In Progress,Charlie,2025-10-17,2025-10-18,Routine inspection,1.75,Yes,No,No,No,No
55,2024-12-22,1009,Pressure drop,Low,Closed,David,2025-10-18,2025-10-19,Adjusted pressure,0.44,No,No,No,No,No
56,2024-12-23,1005,Overheating,Medium,In Progress,Bob,2025-10-19,2025-10-20,Routine inspection,0.39,No,No,No,No,No
57,2024-12-24,1001,Leakage,Medium,Closed,Alice,2025-10-20,2025-10-21,Replaced faulty valve,0.48,No,No,No,No,No
58,2024-12-25,1003,Pressure drop,Low,Open,Bob,2025-10-21,2025-10-22,Adjusted pressure,0.39,No,No,No,No,No
59,2024-12-26,1004,Overheating,Medium,InProgress,David,2025-10-22,2025-10-23,Routine inspection,0.34,No,No,No,No,No

60,2024-12-27,1007,Leakage,High,Closed,Alice,2025-10-23,2025-10-24,Replaced faulty valve,0.67,No,No,No,No,No"""

api_df = pd.read_csv(StringIO(api_data_str))
maintenance_df = pd.read_csv(StringIO(maintenance_str))

# -------------------------
# Feature Engineering
# -------------------------
def create_features(api_df, maintenance_df):
    # Aggregate sensor features per equipment
    feats = api_df.groupby("equipment_id").agg({
        "temperature": ["mean","max","std"],
        "vibration": ["mean","max","std"],
        "pressure": ["mean","min","std"],
        "anomaly_score": "mean",
        "predicted_failure_in_days": "mean"
    })
    feats.columns = ["_".join(c) for c in feats.columns]
    feats = feats.reset_index()

    # Labels: last known maintenance actions
    labels = maintenance_df[[
        "equipment_id","brakes_replaced","tires_replaced",
        "chain_replaced","gears_replaced","electronics_replaced"
    ]].replace({"Yes":1,"No":0}).groupby("equipment_id").max().reset_index()

    df = pd.merge(feats, labels, on="equipment_id", how="left").fillna(0)
    return df

df = create_features(api_df, maintenance_df)

X = df.drop(columns=["equipment_id","brakes_replaced","tires_replaced","chain_replaced","gears_replaced","electronics_replaced"])
y = df[["brakes_replaced","tires_replaced","chain_replaced","gears_replaced","electronics_replaced"]]

# -------------------------
# PyTorch Model
# -------------------------
class MLPModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers(x)

# -------------------------
# Train Function
# -------------------------
def train_model(X, y, epochs=100, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).to(device)

    X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    model = MLPModel(X.shape[1], y.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val)
                val_loss = criterion(val_preds, y_val).item()
                print(f"Epoch {epoch}: Train Loss {loss.item():.4f}, Val Loss {val_loss:.4f}")

    return model

# -------------------------
# Streamlit UI
# -------------------------
st.title("⚙️ Equipment Failure Prediction Dashboard")

if st.button("Train & Predict"):
    with st.spinner("Training model..."):
        model = train_model(X, y, epochs=50)

    # Predictions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        preds = model(torch.tensor(X.values, dtype=torch.float32).to(device)).cpu().numpy()

    predictions_df = pd.DataFrame(preds, columns=y.columns)
    predictions_df["equipment_id"] = df["equipment_id"]

    st.success("✅ Training Complete! Here are the predictions:")
    st.dataframe(predictions_df.style.background_gradient(cmap="coolwarm", axis=0))
