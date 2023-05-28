from .video_record import VideoRecord


class ActionNetRecord(VideoRecord):
    def __init__(self, tup, dataset_conf):
        self._index = str(tup[0])
        self._series = tup[1]
        self.dataset_conf = dataset_conf

    @property
    def start_frame(self):
        return self._series['start_frame'] - 1

    @property
    def end_frame(self):
        return self._series['stop_frame'] - 2

    @property
    def uid(self):
        return self._series['uid']

    @property
    def untrimmed_video_name(self):
        return self._series['subject']
        

    @property
    def label(self):
        if 'verb_class' not in self._series.keys().tolist():
            raise NotImplementedError
        return self._series['verb_class']

    @property
    def myo_left_readings(self):
        return self._series['myo_left_readings']
    
    @property
    def myo_right_readings(self):
        return self._series['myo_right_readings']
    
    @property
    def num_frames(self):
        return {'RGB': self.end_frame - self.start_frame,
                'EMG': len(self._series['myo_right_readings'])//2}