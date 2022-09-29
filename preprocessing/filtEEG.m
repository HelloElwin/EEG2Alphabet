function eeg = filtEEG(eeg, filtRng, filtOrder, fs)
% filtRng = [freqLow, freqHigh];
% filtOrder

eegData = eeg.data; % ch x points x trials
[nCh, nPoints, nTrials] = size(eegData);
eegData = permute(eegData, [2 1 3]); % points x ch x trials
eegData = double(eegData(:,:)); % points x (ch x trials)

Wn_full = [filtRng(1)*2 filtRng(2)*2]/fs;%band pass:lowfs:highfs;
[k_full,l_full] = butter(filtOrder,Wn_full);
eegData_filtered = filtfilt(k_full,l_full, eegData);

eegData_filtered = reshape(eegData_filtered, nPoints, nCh, []);
eegData_filtered = permute(eegData_filtered, [2 1 3]);

eeg.data = eegData_filtered;
