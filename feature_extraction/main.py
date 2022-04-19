from extract_features import Features

feature_extractor = Features()
feature_extractor()




wavfile = '/worktmp2/hxkhkh/current/Dcase/data/clotho/audio/validation/51_STRWA.wav'

import librosa
y, sr = librosa.load(wavfile)