# Copyright (c) 2024, Zhendong Peng (pzd17@tsinghua.org.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from modelscope import model_file_download, snapshot_download
from sherpa_onnx import (
    FastClusteringConfig,
    OfflineRecognizer,
    OfflineSpeakerDiarization,
    OfflineSpeakerDiarizationConfig,
    OfflineSpeakerSegmentationModelConfig,
    OfflineSpeakerSegmentationPyannoteModelConfig,
    SpeakerEmbeddingExtractorConfig,
)
from tqdm.auto import tqdm


class SpeakerDiarization:
    def __init__(
        self,
        segmentation_model="pengzhendong/sherpa-onnx-pyannote-segmentation-3-0",
        embedding_extractor_model="3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
        num_speakers=-1,
        cluster_threshold=0.75,
        asr_model="pengzhendong/sherpa-onnx-sense-voice-zh-en-ja-ko-yue",
        language="auto",
        use_itn=False,
    ):
        """
        Args:
        num_speakers:
            If you know the actual number of speakers in the wave file, then please
            specify it. Otherwise, leave it to -1
        cluster_threshold:
            If num_speakers is -1, then this threshold is used for clustering.
            A smaller cluster_threshold leads to more clusters, i.e., more speakers.
            A larger cluster_threshold leads to fewer clusters, i.e., fewer speakers.
        language:
            If not empty, then valid values are: auto, zh, en, ja, ko, yue
        use_itn:
            True to enable inverse text normalization; False to disable it.
        """
        segmentation_model = f"{snapshot_download(segmentation_model)}/model.onnx"
        embedding_extractor_model = model_file_download(
            model_id="pengzhendong/speaker-identification",
            file_path=embedding_extractor_model,
        )
        config = OfflineSpeakerDiarizationConfig(
            segmentation=OfflineSpeakerSegmentationModelConfig(
                pyannote=OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=segmentation_model
                ),
            ),
            embedding=SpeakerEmbeddingExtractorConfig(model=embedding_extractor_model),
            clustering=FastClusteringConfig(
                num_clusters=num_speakers, threshold=cluster_threshold
            ),
            min_duration_on=0.3,
            min_duration_off=0.5,
        )
        if not config.validate():
            raise RuntimeError(
                "Please check your config and make sure all required files exist"
            )
        self.diarizer = OfflineSpeakerDiarization(config)

        asr_model = snapshot_download(asr_model)
        self.recognizer = OfflineRecognizer.from_sense_voice(
            model=f"{asr_model}/model.onnx",
            tokens=f"{asr_model}/tokens.txt",
            language=language,
            use_itn=use_itn,
        )

    @staticmethod
    def progress_callback(num_processed_chunk, num_total_chunks, progress_bar=None):
        progress_bar.total = num_total_chunks
        progress_bar.update(num_processed_chunk - progress_bar.n)
        if num_processed_chunk == num_total_chunks:
            progress_bar.close()
        return 0

    def process(self, audio, rate=None, show_progress=True, progress_callback=None):
        if isinstance(audio, str):
            audio = Path(audio)
        if isinstance(audio, Path):
            if not audio.exists():
                raise FileNotFoundError(f"The file {audio} does not exist.")
            audio, rate = sf.read(audio, dtype="float32", always_2d=True)
        elif isinstance(audio, np.ndarray):
            if rate is None:
                raise ValueError("When audio is an ndarray, rate must be provided.")
        else:
            raise TypeError("Input must be a string, Path, or numpy ndarray.")
        if rate != self.diarizer.sample_rate:
            audio = librosa.resample(
                audio, orig_sr=rate, target_sr=self.diarizer.sample_rate, axis=0
            )
            rate = self.diarizer.sample_rate

        if show_progress:
            if progress_callback is None:
                progress_callback = partial(
                    self.progress_callback,
                    progress_bar=tqdm(desc="Progress", unit="chunks"),
                )
            result = self.diarizer.process(audio, callback=progress_callback)
        else:
            result = self.diarizer.process(audio)
        for r in result.sort_by_start_time():
            stream = self.recognizer.create_stream()
            segment = audio[int(r.start * rate) : int(r.end * rate), :]
            stream.accept_waveform(rate, segment)
            self.recognizer.decode_stream(stream)

            yield {
                "start": round(r.start, 3),
                "duration": round(r.end - r.start, 3),
                "speaker": f"speaker_{r.speaker:02}",
                "lang": stream.result.lang,
                "emotion": stream.result.emotion,
                "event": stream.result.event,
                "text": stream.result.text,
                "tokens": stream.result.tokens,
                "timestamps": [
                    round(timestamp, 3) for timestamp in stream.result.timestamps
                ],
            }
