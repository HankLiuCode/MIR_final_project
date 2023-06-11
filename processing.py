class SpectrogramProcessor:
    def __init__(self, spectrogram, timestamps, annotations_file):
        self.spectrogram = spectrogram
        self.timestamps = timestamps
        self.annotations_file = annotations_file

    def get_annotations(self):
        with open(self.annotations_file) as file:
            annotations = []
            for line in file:
                if line != "\n":
                    annotations.append(float(line.rstrip()))
        return annotations

    def split_spectrogram(self, split_size):
        spectrogram = self.spectrogram
        timestamps = self.timestamps
        spectrograms = []

        for i in range(len(timestamps)):
            time = timestamps[i]
            s = spectrogram[:, i: i + split_size]
            if s.shape[1] == split_size:
                spectrograms.append((s, time))
        return spectrograms

    def get_onsets(self, spectrograms, annotations, diff_from_onset_ms):
        onsets = []
        if not annotations:
            onsets = [0] * len(spectrograms)
            return onsets

        current_index = 0
        for annotation in annotations:
            for i in range(current_index, len(spectrograms)):
                spectrogram, time = spectrograms[i]

                if ((annotation - diff_from_onset_ms) <= time) and (time <= annotation):
                    onsets.append(1)

                    # If we are at the last annotation, don't come out of the
                    # inner loop before finishing.
                    if annotation == annotations[-1]:
                        continue
                elif ((annotation + diff_from_onset_ms) < time) and (
                    annotation != annotations[-1]
                ):
                    onsets.append(0)
                    break
                else:
                    onsets.append(0)

            current_index = i + 1
        assert len(spectrograms) == len(onsets)
        return onsets
