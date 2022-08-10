import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from more_itertools import sample
import numpy as np
import pandas as pd

# sample="background_sound/91.wav"
# data,sample_rate=librosa.load(sample)

# plt.title("Waveform")
# librosa.display.waveshow(data,sr=sample_rate)
# plt.show()

# mfccs=librosa.feature.mfcc(y=data,sr=sample_rate,n_mfcc=40)
# print("Shape of mfcc : ",mfccs.shape)

# plt.title("MFCC")
# librosa.display.specshow(mfccs,sr=sample_rate,x_axis='time')
# plt.show()

all_data=[]
data_path_dict={
    0:["background_sounds/"+file_path for file_path in os.listdir("background_sounds/")],
    1:["Trigger_word/"+file_path for file_path in os.listdir("Trigger_word/")]
}

for class_labels,list_of_files in data_path_dict.items():
    for single_file in list_of_files:
        data,sample_rate=librosa.load(single_file)
        mfccs=librosa.feature.mfcc(y=data,sr=sample_rate,n_mfcc=40)
        mfcc_processed=np.mean(mfccs.T,axis=0)
        all_data.append([mfcc_processed,class_labels])
    print(f"Succesfully preprocessed class label {class_labels}")

df=pd.DataFrame(all_data,columns=["features","class"])
df.to_pickle("final_audio_data_csv/audio_data.csv")