clear

load('datas.mat');
%%
eeg.data = data(:, 1:801, 1:500);

filtered_12_15_beta = filtEEG(eeg, [12.0, 15], 2, 500);
filtered_4_8_theta = filtEEG(eeg, [4, 8], 2, 250);
filtered_8_12_beta = filtEEG(eeg, [8, 12], 2, 250);
filtered_2_4_beta = filtEEG(eeg, [2, 4], 2, 250);


save filtered_data.mat filtered_12_15_beta filtered_4_8_theta filtered_8_12_beta filtered_2_4_beta