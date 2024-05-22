{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "85471364",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting streamlit\n",
      "  Downloading streamlit-1.34.0-py2.py3-none-any.whl.metadata (8.5 kB)\n",
      "Requirement already satisfied: altair<6,>=4.0 in c:\\users\\alpin limited\\anaconda3\\envs\\my_env\\lib\\site-packages (from streamlit) (5.3.0)\n",
      "Collecting blinker<2,>=1.0.0 (from streamlit)\n",
      "  Downloading blinker-1.8.2-py3-none-any.whl.metadata (1.6 kB)\n",
      "Collecting cachetools<6,>=4.0 (from streamlit)\n",
      "  Downloading cachetools-5.3.3-py3-none-any.whl.metadata (5.3 kB)\n",
      "Requirement already satisfied: click<9,>=7.0 in c:\\users\\alpin limited\\anaconda3\\envs\\my_env\\lib\\site-packages (from streamlit) (8.1.7)\n",
      "Requirement already satisfied: numpy<2,>=1.19.3 in c:\\users\\alpin limited\\anaconda3\\envs\\my_env\\lib\\site-packages (from streamlit) (1.26.4)\n",
      "Requirement already satisfied: packaging<25,>=16.8 in c:\\users\\alpin limited\\anaconda3\\envs\\my_env\\lib\\site-packages (from streamlit) (23.2)\n",
      "Requirement already satisfied: pandas<3,>=1.3.0 in c:\\users\\alpin limited\\anaconda3\\envs\\my_env\\lib\\site-packages (from streamlit) (2.2.2)\n",
      "Requirement already satisfied: pillow<11,>=7.1.0 in c:\\users\\alpin limited\\anaconda3\\envs\\my_env\\lib\\site-packages (from streamlit) (10.3.0)\n",
      "Collecting protobuf<5,>=3.20 (from streamlit)\n",
      "  Downloading protobuf-4.25.3-cp39-cp39-win_amd64.whl.metadata (541 bytes)\n",
      "Collecting pyarrow>=7.0 (from streamlit)\n",
      "  Downloading pyarrow-16.1.0-cp39-cp39-win_amd64.whl.metadata (3.1 kB)\n",
      "Requirement already satisfied: requests<3,>=2.27 in c:\\users\\alpin limited\\anaconda3\\envs\\my_env\\lib\\site-packages (from streamlit) (2.31.0)\n",
      "Requirement already satisfied: rich<14,>=10.14.0 in c:\\users\\alpin limited\\anaconda3\\envs\\my_env\\lib\\site-packages (from streamlit) (13.7.1)\n",
      "Collecting tenacity<9,>=8.1.0 (from streamlit)\n",
      "  Downloading tenacity-8.3.0-py3-none-any.whl.metadata (1.2 kB)\n",
      "Collecting toml<2,>=0.10.1 (from streamlit)\n",
      "  Downloading toml-0.10.2-py2.py3-none-any.whl.metadata (7.1 kB)\n",
      "Requirement already satisfied: typing-extensions<5,>=4.3.0 in c:\\users\\alpin limited\\anaconda3\\envs\\my_env\\lib\\site-packages (from streamlit) (4.11.0)\n",
      "Collecting gitpython!=3.1.19,<4,>=3.0.7 (from streamlit)\n",
      "  Downloading GitPython-3.1.43-py3-none-any.whl.metadata (13 kB)\n",
      "Collecting pydeck<1,>=0.8.0b4 (from streamlit)\n",
      "  Downloading pydeck-0.9.1-py2.py3-none-any.whl.metadata (4.1 kB)\n",
      "Requirement already satisfied: tornado<7,>=6.0.3 in c:\\users\\alpin limited\\anaconda3\\envs\\my_env\\lib\\site-packages (from streamlit) (6.3.3)\n",
      "Collecting watchdog>=2.1.5 (from streamlit)\n",
      "  Downloading watchdog-4.0.0-py3-none-win_amd64.whl.metadata (37 kB)\n",
      "Requirement already satisfied: jinja2 in c:\\users\\alpin limited\\anaconda3\\envs\\my_env\\lib\\site-packages (from altair<6,>=4.0->streamlit) (3.1.3)\n",
      "Requirement already satisfied: jsonschema>=3.0 in c:\\users\\alpin limited\\anaconda3\\envs\\my_env\\lib\\site-packages (from altair<6,>=4.0->streamlit) (4.19.2)\n",
      "Requirement already satisfied: toolz in c:\\users\\alpin limited\\anaconda3\\envs\\my_env\\lib\\site-packages (from altair<6,>=4.0->streamlit) (0.12.1)\n",
      "Requirement already satisfied: colorama in c:\\users\\alpin limited\\anaconda3\\envs\\my_env\\lib\\site-packages (from click<9,>=7.0->streamlit) (0.4.6)\n",
      "Collecting gitdb<5,>=4.0.1 (from gitpython!=3.1.19,<4,>=3.0.7->streamlit)\n",
      "  Downloading gitdb-4.0.11-py3-none-any.whl.metadata (1.2 kB)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in c:\\users\\alpin limited\\anaconda3\\envs\\my_env\\lib\\site-packages (from pandas<3,>=1.3.0->streamlit) (2.8.2)\n",
      "Requirement already satisfied: pytz>=2020.1 in c:\\users\\alpin limited\\anaconda3\\envs\\my_env\\lib\\site-packages (from pandas<3,>=1.3.0->streamlit) (2024.1)\n",
      "Requirement already satisfied: tzdata>=2022.7 in c:\\users\\alpin limited\\anaconda3\\envs\\my_env\\lib\\site-packages (from pandas<3,>=1.3.0->streamlit) (2024.1)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in c:\\users\\alpin limited\\anaconda3\\envs\\my_env\\lib\\site-packages (from requests<3,>=2.27->streamlit) (2.0.4)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\users\\alpin limited\\anaconda3\\envs\\my_env\\lib\\site-packages (from requests<3,>=2.27->streamlit) (3.7)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in c:\\users\\alpin limited\\anaconda3\\envs\\my_env\\lib\\site-packages (from requests<3,>=2.27->streamlit) (2.2.1)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\alpin limited\\anaconda3\\envs\\my_env\\lib\\site-packages (from requests<3,>=2.27->streamlit) (2024.2.2)\n",
      "Requirement already satisfied: markdown-it-py>=2.2.0 in c:\\users\\alpin limited\\anaconda3\\envs\\my_env\\lib\\site-packages (from rich<14,>=10.14.0->streamlit) (3.0.0)\n",
      "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in c:\\users\\alpin limited\\anaconda3\\envs\\my_env\\lib\\site-packages (from rich<14,>=10.14.0->streamlit) (2.15.1)\n",
      "Collecting smmap<6,>=3.0.1 (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit)\n",
      "  Downloading smmap-5.0.1-py3-none-any.whl.metadata (4.3 kB)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in c:\\users\\alpin limited\\anaconda3\\envs\\my_env\\lib\\site-packages (from jinja2->altair<6,>=4.0->streamlit) (2.1.3)\n",
      "Requirement already satisfied: attrs>=22.2.0 in c:\\users\\alpin limited\\anaconda3\\envs\\my_env\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (23.1.0)\n",
      "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in c:\\users\\alpin limited\\anaconda3\\envs\\my_env\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2023.12.1)\n",
      "Requirement already satisfied: referencing>=0.28.4 in c:\\users\\alpin limited\\anaconda3\\envs\\my_env\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.30.2)\n",
      "Requirement already satisfied: rpds-py>=0.7.1 in c:\\users\\alpin limited\\anaconda3\\envs\\my_env\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.10.6)\n",
      "Requirement already satisfied: mdurl~=0.1 in c:\\users\\alpin limited\\anaconda3\\envs\\my_env\\lib\\site-packages (from markdown-it-py>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.2)\n",
      "Requirement already satisfied: six>=1.5 in c:\\users\\alpin limited\\anaconda3\\envs\\my_env\\lib\\site-packages (from python-dateutil>=2.8.2->pandas<3,>=1.3.0->streamlit) (1.16.0)\n",
      "Collecting referencing>=0.28.4 (from jsonschema>=3.0->altair<6,>=4.0->streamlit)\n",
      "  Using cached referencing-0.35.1-py3-none-any.whl.metadata (2.8 kB)\n",
      "Downloading streamlit-1.34.0-py2.py3-none-any.whl (8.5 MB)\n",
      "   ---------------------------------------- 0.0/8.5 MB ? eta -:--:--\n",
      "    --------------------------------------- 0.2/8.5 MB 5.4 MB/s eta 0:00:02\n",
      "   ---- ----------------------------------- 0.9/8.5 MB 11.3 MB/s eta 0:00:01\n",
      "   ------ --------------------------------- 1.4/8.5 MB 11.3 MB/s eta 0:00:01\n",
      "   ---------- ----------------------------- 2.2/8.5 MB 12.4 MB/s eta 0:00:01\n",
      "   -------------- ------------------------- 3.0/8.5 MB 13.6 MB/s eta 0:00:01\n",
      "   ---------------- ----------------------- 3.5/8.5 MB 12.9 MB/s eta 0:00:01\n",
      "   ------------------- -------------------- 4.1/8.5 MB 13.0 MB/s eta 0:00:01\n",
      "   ---------------------- ----------------- 4.9/8.5 MB 13.5 MB/s eta 0:00:01\n",
      "   ------------------------- -------------- 5.5/8.5 MB 14.0 MB/s eta 0:00:01\n",
      "   ---------------------------- ----------- 6.1/8.5 MB 14.0 MB/s eta 0:00:01\n",
      "   -------------------------------- ------- 6.8/8.5 MB 14.1 MB/s eta 0:00:01\n",
      "   ----------------------------------- ---- 7.5/8.5 MB 14.5 MB/s eta 0:00:01\n",
      "   -------------------------------------- - 8.3/8.5 MB 14.3 MB/s eta 0:00:01\n",
      "   ---------------------------------------  8.5/8.5 MB 13.9 MB/s eta 0:00:01\n",
      "   ---------------------------------------  8.5/8.5 MB 13.9 MB/s eta 0:00:01\n",
      "   ---------------------------------------  8.5/8.5 MB 13.9 MB/s eta 0:00:01\n",
      "   ---------------------------------------  8.5/8.5 MB 13.9 MB/s eta 0:00:01\n",
      "   ---------------------------------------- 8.5/8.5 MB 10.7 MB/s eta 0:00:00\n",
      "Downloading blinker-1.8.2-py3-none-any.whl (9.5 kB)\n",
      "Downloading cachetools-5.3.3-py3-none-any.whl (9.3 kB)\n",
      "Downloading GitPython-3.1.43-py3-none-any.whl (207 kB)\n",
      "   ---------------------------------------- 0.0/207.3 kB ? eta -:--:--\n",
      "   ---------------------------------------- 207.3/207.3 kB 6.4 MB/s eta 0:00:00\n",
      "Downloading protobuf-4.25.3-cp39-cp39-win_amd64.whl (413 kB)\n",
      "   ---------------------------------------- 0.0/413.4 kB ? eta -:--:--\n",
      "   --------------------------------------- 413.4/413.4 kB 25.2 MB/s eta 0:00:00\n",
      "Downloading pyarrow-16.1.0-cp39-cp39-win_amd64.whl (25.9 MB)\n",
      "   ---------------------------------------- 0.0/25.9 MB ? eta -:--:--\n",
      "   - -------------------------------------- 0.8/25.9 MB 16.1 MB/s eta 0:00:02\n",
      "   - -------------------------------------- 1.3/25.9 MB 16.1 MB/s eta 0:00:02\n",
      "   --- ------------------------------------ 2.0/25.9 MB 14.4 MB/s eta 0:00:02\n",
      "   ---- ----------------------------------- 2.8/25.9 MB 16.3 MB/s eta 0:00:02\n",
      "   ------ --------------------------------- 4.0/25.9 MB 16.9 MB/s eta 0:00:02\n",
      "   ------- -------------------------------- 4.7/25.9 MB 17.5 MB/s eta 0:00:02\n",
      "   ------- -------------------------------- 4.9/25.9 MB 16.5 MB/s eta 0:00:02\n",
      "   --------- ------------------------------ 5.9/25.9 MB 16.5 MB/s eta 0:00:02\n",
      "   --------- ------------------------------ 6.4/25.9 MB 15.7 MB/s eta 0:00:02\n",
      "   ---------- ----------------------------- 7.0/25.9 MB 16.1 MB/s eta 0:00:02\n",
      "   ----------- ---------------------------- 7.5/25.9 MB 15.6 MB/s eta 0:00:02\n",
      "   ------------ --------------------------- 8.4/25.9 MB 15.7 MB/s eta 0:00:02\n",
      "   -------------- ------------------------- 9.1/25.9 MB 16.2 MB/s eta 0:00:02\n",
      "   -------------- ------------------------- 9.5/25.9 MB 15.7 MB/s eta 0:00:02\n",
      "   --------------- ------------------------ 10.1/25.9 MB 15.8 MB/s eta 0:00:01\n",
      "   ---------------- ----------------------- 10.9/25.9 MB 15.6 MB/s eta 0:00:01\n",
      "   ----------------- ---------------------- 11.5/25.9 MB 16.4 MB/s eta 0:00:01\n",
      "   ----------------- ---------------------- 11.5/25.9 MB 16.4 MB/s eta 0:00:01\n",
      "   ----------------- ---------------------- 11.5/25.9 MB 16.4 MB/s eta 0:00:01\n",
      "   ----------------- ---------------------- 11.5/25.9 MB 16.4 MB/s eta 0:00:01\n",
      "   ----------------- ---------------------- 11.5/25.9 MB 16.4 MB/s eta 0:00:01\n",
      "   ----------------- ---------------------- 11.5/25.9 MB 16.4 MB/s eta 0:00:01\n",
      "   ----------------- ---------------------- 11.6/25.9 MB 11.5 MB/s eta 0:00:02\n",
      "   ------------------ --------------------- 12.2/25.9 MB 11.5 MB/s eta 0:00:02\n",
      "   -------------------- ------------------- 13.0/25.9 MB 11.3 MB/s eta 0:00:02\n",
      "   --------------------- ------------------ 13.6/25.9 MB 11.3 MB/s eta 0:00:02\n",
      "   --------------------- ------------------ 14.2/25.9 MB 11.1 MB/s eta 0:00:02\n",
      "   ---------------------- ----------------- 14.9/25.9 MB 10.9 MB/s eta 0:00:02\n",
      "   ------------------------ --------------- 15.6/25.9 MB 11.3 MB/s eta 0:00:01\n",
      "   ------------------------ --------------- 15.7/25.9 MB 11.3 MB/s eta 0:00:01\n",
      "   ------------------------ --------------- 15.8/25.9 MB 10.6 MB/s eta 0:00:01\n",
      "   -------------------------- ------------- 17.0/25.9 MB 10.7 MB/s eta 0:00:01\n",
      "   --------------------------- ------------ 17.9/25.9 MB 11.1 MB/s eta 0:00:01\n",
      "   ----------------------------- ---------- 18.8/25.9 MB 11.1 MB/s eta 0:00:01\n",
      "   ------------------------------ --------- 19.5/25.9 MB 11.1 MB/s eta 0:00:01\n",
      "   ------------------------------ --------- 19.9/25.9 MB 11.3 MB/s eta 0:00:01\n",
      "   ------------------------------ --------- 19.9/25.9 MB 11.3 MB/s eta 0:00:01\n",
      "   ------------------------------ --------- 19.9/25.9 MB 11.3 MB/s eta 0:00:01\n",
      "   ------------------------------ --------- 19.9/25.9 MB 11.3 MB/s eta 0:00:01\n",
      "   ------------------------------ --------- 19.9/25.9 MB 11.3 MB/s eta 0:00:01\n",
      "   ------------------------------ --------- 19.9/25.9 MB 11.3 MB/s eta 0:00:01\n",
      "   ------------------------------- -------- 20.2/25.9 MB 8.7 MB/s eta 0:00:01\n",
      "   ------------------------------- -------- 20.6/25.9 MB 8.6 MB/s eta 0:00:01\n",
      "   -------------------------------- ------- 21.3/25.9 MB 8.5 MB/s eta 0:00:01\n",
      "   --------------------------------- ------ 21.7/25.9 MB 8.4 MB/s eta 0:00:01\n",
      "   ---------------------------------- ----- 22.2/25.9 MB 10.6 MB/s eta 0:00:01\n",
      "   ----------------------------------- ---- 22.7/25.9 MB 10.2 MB/s eta 0:00:01\n",
      "   ----------------------------------- ---- 23.1/25.9 MB 10.1 MB/s eta 0:00:01\n",
      "   ------------------------------------ --- 23.7/25.9 MB 10.1 MB/s eta 0:00:01\n",
      "   ------------------------------------- -- 24.3/25.9 MB 10.1 MB/s eta 0:00:01\n",
      "   -------------------------------------- - 25.1/25.9 MB 10.2 MB/s eta 0:00:01\n",
      "   ---------------------------------------  25.8/25.9 MB 10.1 MB/s eta 0:00:01\n",
      "   ---------------------------------------  25.9/25.9 MB 10.1 MB/s eta 0:00:01\n",
      "   ---------------------------------------  25.9/25.9 MB 10.1 MB/s eta 0:00:01\n",
      "   ---------------------------------------  25.9/25.9 MB 10.1 MB/s eta 0:00:01\n",
      "   ---------------------------------------  25.9/25.9 MB 10.1 MB/s eta 0:00:01\n",
      "   ---------------------------------------  25.9/25.9 MB 10.1 MB/s eta 0:00:01\n",
      "   ---------------------------------------  25.9/25.9 MB 10.1 MB/s eta 0:00:01\n",
      "   ---------------------------------------  25.9/25.9 MB 10.1 MB/s eta 0:00:01\n",
      "   ---------------------------------------  25.9/25.9 MB 10.1 MB/s eta 0:00:01\n",
      "   ---------------------------------------  25.9/25.9 MB 10.1 MB/s eta 0:00:01\n",
      "   ---------------------------------------  25.9/25.9 MB 10.1 MB/s eta 0:00:01\n",
      "   ---------------------------------------  25.9/25.9 MB 10.1 MB/s eta 0:00:01\n",
      "   ---------------------------------------  25.9/25.9 MB 10.1 MB/s eta 0:00:01\n",
      "   ---------------------------------------  25.9/25.9 MB 10.1 MB/s eta 0:00:01\n",
      "   ---------------------------------------  25.9/25.9 MB 10.1 MB/s eta 0:00:01\n",
      "   ---------------------------------------  25.9/25.9 MB 10.1 MB/s eta 0:00:01\n",
      "   ---------------------------------------  25.9/25.9 MB 10.1 MB/s eta 0:00:01\n",
      "   ---------------------------------------  25.9/25.9 MB 10.1 MB/s eta 0:00:01\n",
      "   ---------------------------------------- 25.9/25.9 MB 5.9 MB/s eta 0:00:00\n",
      "Downloading pydeck-0.9.1-py2.py3-none-any.whl (6.9 MB)\n",
      "   ---------------------------------------- 0.0/6.9 MB ? eta -:--:--\n",
      "   ---- ----------------------------------- 0.9/6.9 MB 26.6 MB/s eta 0:00:01\n",
      "   --------- ------------------------------ 1.7/6.9 MB 21.4 MB/s eta 0:00:01\n",
      "   ------------- -------------------------- 2.3/6.9 MB 18.4 MB/s eta 0:00:01\n",
      "   ------------------- -------------------- 3.3/6.9 MB 19.2 MB/s eta 0:00:01\n",
      "   ----------------------- ---------------- 4.1/6.9 MB 18.7 MB/s eta 0:00:01\n",
      "   --------------------------- ------------ 4.7/6.9 MB 18.8 MB/s eta 0:00:01\n",
      "   ------------------------------- -------- 5.4/6.9 MB 18.0 MB/s eta 0:00:01\n",
      "   ------------------------------------- -- 6.5/6.9 MB 18.8 MB/s eta 0:00:01\n",
      "   ---------------------------------------  6.9/6.9 MB 17.6 MB/s eta 0:00:01\n",
      "   ---------------------------------------  6.9/6.9 MB 17.6 MB/s eta 0:00:01\n",
      "   ---------------------------------------  6.9/6.9 MB 17.6 MB/s eta 0:00:01\n",
      "   ---------------------------------------  6.9/6.9 MB 17.6 MB/s eta 0:00:01\n",
      "   ---------------------------------------  6.9/6.9 MB 17.6 MB/s eta 0:00:01\n",
      "   ---------------------------------------  6.9/6.9 MB 17.6 MB/s eta 0:00:01\n",
      "   ---------------------------------------- 6.9/6.9 MB 9.8 MB/s eta 0:00:00\n",
      "Downloading tenacity-8.3.0-py3-none-any.whl (25 kB)\n",
      "Downloading toml-0.10.2-py2.py3-none-any.whl (16 kB)\n",
      "Downloading watchdog-4.0.0-py3-none-win_amd64.whl (82 kB)\n",
      "   ---------------------------------------- 0.0/82.9 kB ? eta -:--:--\n",
      "   ---------------------------------------- 82.9/82.9 kB 4.8 MB/s eta 0:00:00\n",
      "Downloading gitdb-4.0.11-py3-none-any.whl (62 kB)\n",
      "   ---------------------------------------- 0.0/62.7 kB ? eta -:--:--\n",
      "   ---------------------------------------- 62.7/62.7 kB 1.7 MB/s eta 0:00:00\n",
      "Using cached referencing-0.35.1-py3-none-any.whl (26 kB)\n",
      "Downloading smmap-5.0.1-py3-none-any.whl (24 kB)\n",
      "Installing collected packages: watchdog, toml, tenacity, smmap, referencing, pyarrow, protobuf, cachetools, blinker, pydeck, gitdb, gitpython, streamlit\n",
      "  Attempting uninstall: referencing\n",
      "    Found existing installation: referencing 0.30.2\n",
      "    Uninstalling referencing-0.30.2:\n",
      "      Successfully uninstalled referencing-0.30.2\n",
      "Successfully installed blinker-1.8.2 cachetools-5.3.3 gitdb-4.0.11 gitpython-3.1.43 protobuf-4.25.3 pyarrow-16.1.0 pydeck-0.9.1 referencing-0.35.1 smmap-5.0.1 streamlit-1.34.0 tenacity-8.3.0 toml-0.10.2 watchdog-4.0.0\n"
     ]
    }
   ],
   "source": [
    "!pip install streamlit"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "34b897a8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pickle\n",
    "import numpy as np\n",
    "import xgboost as xgb"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "63522fa5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the model using the new version\n",
    "model = xgb.Booster()\n",
    "model.load_model('model.json')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "35e00bfe-2c7a-46ae-a77e-9fae675f4606",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define the prediction function\n",
    "def predict_performance(design_flow, actual_flow):\n",
    "    # Create a DataFrame with the input values\n",
    "    input_df = pd.DataFrame({'Design_flow(l/s)': [design_flow], 'Actual_flow(l/s)': [actual_flow]})\n",
    "    # Use the model to make a prediction\n",
    "    prediction = loaded_model.predict(input_df)\n",
    "    # Round the prediction\n",
    "    rounded_prediction = np.round(prediction)\n",
    "    return rounded_prediction[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "acba97b1-45c7-43bd-b36e-fa00b0536f66",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-05-22 10:11:14.067 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\Alpin Limited\\anaconda3\\envs\\my_env\\lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2024-05-22 10:11:14.084 Session state does not function when running a script without `streamlit run`\n"
     ]
    }
   ],
   "source": [
    "# Define the Streamlit app\n",
    "def main():\n",
    "    st.title(\"Machine Learning Model Prediction Interface\")\n",
    "    \n",
    "    # Create input fields for the features expected by the model\n",
    "    # Replace 'feature1', 'feature2', etc. with actual feature names or descriptions\n",
    "    feature1 = st.number_input(\"Enter value for feature1:\")\n",
    "    feature2 = st.number_input(\"Enter value for feature2:\")\n",
    "# Streamlit app\n",
    "def main():\n",
    "    st.title(\"FCU Performance Prediction\")\n",
    "    \n",
    "    st.write(\"\"\"\n",
    "    ## Enter the design and actual flow values to predict the performance percentage:\n",
    "    \"\"\")\n",
    "    \n",
    "    design_flow = st.number_input(\"Design Flow (l/s)\", min_value=0.0, step=0.1, format=\"%.2f\")\n",
    "    actual_flow = st.number_input(\"Actual Flow (l/s)\", min_value=0.0, step=0.1, format=\"%.2f\")\n",
    "    \n",
    "    if st.button(\"Predict\"):\n",
    "        result = predict_performance(design_flow, actual_flow)\n",
    "        st.write(f\"Predicted Performance (%): {result}\")\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main()\n"
   ]
  },
  {
   "cell_type": "code",
   "id": "f892ac34",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.19"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
