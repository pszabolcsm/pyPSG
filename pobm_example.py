import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pobm.obm.desat import DesaturationsMeasures, desat_embedding
from pobm.prep import set_range, median_spo2

def features_all_desat(signal, time_signal, ODI_Threshold = 6, hard_threshold = 88, relative = True, desat_max_length = 14400):
    time_signal = np.array(time_signal)

    # desat_class = DesaturationsMeasures(ODI_Threshold=ODI_Threshold,hard_threshold=hard_threshold, relative=relative, desat_max_length=desat_max_length)
    desat_class = DesaturationsMeasures(ODI_Threshold=ODI_Threshold, hard_threshold=hard_threshold, desat_max_length=desat_max_length)
    desat_class.compute(signal)

    begin_idx = desat_class.begin
    end_idx = desat_class.end

    desaturations, desaturation_valid, desaturation_length_all, desaturation_int_100_all, \
    desaturation_int_max_all, desaturation_depth_100_all, desaturation_depth_max_all, \
    desaturation_slope_all = desat_embedding(begin_idx, end_idx)
    time_spo2_array = np.array(range(len(signal)))

    starts = []
    for (i, desaturation) in enumerate(desaturations):
        starts.append(desaturation['Start'])
        desaturation_idx = (time_spo2_array >= desaturation['Start']) & (time_spo2_array <= desaturation['End'])

        if np.sum(desaturation_idx) == 0:
            continue
        signal = np.array(signal)

        desaturation_time = time_spo2_array[desaturation_idx]
        desaturation_spo2 = signal[desaturation_idx]
        desaturation_min = np.nanmin(desaturation_spo2)
        desaturation_max = np.nanmax(desaturation_spo2)

        desaturation_valid[i] = True
        desaturation_length_all[i] = desaturation['Duration']
        desaturation_int_100_all[i] = np.nansum(100 - desaturation_spo2)
        desaturation_int_max_all[i] = np.nansum(desaturation_max - desaturation_spo2)
        desaturation_depth_100_all[i] = 100 - desaturation_min
        desaturation_depth_max_all[i] = desaturation_max - desaturation_min

        desaturation_idx_max = np.where(desaturation_spo2 == desaturation_max)[0][0]
        desaturation_idx_min = np.where(desaturation_spo2 == desaturation_min)[0][-1]
        desaturation_idx_max_min = np.arange(desaturation_idx_max, desaturation_idx_min + 1)

        if len(desaturation_idx_max_min) > 0:
            p = np.polyfit(np.int64(desaturation_time[desaturation_idx_max_min]),
                           desaturation_spo2[desaturation_idx_max_min], 1)

            desaturation_slope_all[i] = p[0]

    begin_time = time_signal[begin_idx]
    end_time = time_signal[end_idx]

    desat_patient = pd.DataFrame({
        "begin": begin_time,
        "end": end_time,
        "begin_idx": begin_idx,
        "end_idx": end_idx,
        "depth": desaturation_depth_max_all,
        "length": desaturation_length_all,
        "area": desaturation_int_max_all
    })
    return desat_patient

pd.read_table('ICU_COVID_patient1.csv',delimiter=',')

data_icu=pd.read_table('ICU_COVID_patient1.csv',delimiter=',')
data_icu['Time'] = pd.to_datetime(data_icu['Time'])
data_icu['SpO2'] = pd.to_numeric(data_icu['SpO2'], errors='coerce')

# # Plot SpO2
# plt.figure(figsize=(5, 3))
# plt.plot(range(0, len(data_icu['SpO2'])), data_icu['SpO2'])
#
# plt.xlabel('Time[s]')
# plt.ylabel('SpO2 (%)')
# plt.title('SpO2 Variation Over Time')
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.show()
#
# # Plot filtered SpO2
spo2_signal= data_icu['SpO2']
spo2_signal = set_range(spo2_signal)
spo2_signal = median_spo2(spo2_signal, FilterLength=301)
#
# plt.figure(figsize=(5, 3))
# plt.plot(range(0, len(spo2_signal)), spo2_signal)
#
# # Plot filtered SpO2
# plt.xlabel('Time[s]')
# plt.ylabel('SpO2 (%)')
# plt.title('SpO2 Variation Over Time')
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.show()

test_desat = features_all_desat(spo2_signal,data_icu['Time'],ODI_Threshold=4, hard_threshold=93, relative=False, desat_max_length =14400)

test_desat