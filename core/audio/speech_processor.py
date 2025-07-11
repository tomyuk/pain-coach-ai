"""Real-time speech processing with faster-whisper."""

import asyncio
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import webrtcvad
from collections import deque
from typing import Callable, Optional
import logging

logger = logging.getLogger(__name__)

class RealTimeSpeechProcessor:
    """Real-time speech recognition optimized for M3 Max."""
    
    def __init__(self, model_size: str = "large-v3"):
        # Initialize Whisper model with M3 Max optimization
        self.whisper_model = WhisperModel(
            model_size, 
            device="auto",  # Auto-detect M3 Max
            compute_type="int8"  # Quantization for speed
        )
        
        # Voice Activity Detection
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level
        
        # Audio buffer
        self.audio_queue = deque(maxlen=30)  # 30 second buffer
        self.is_recording = False
        
        # Configuration
        self.sample_rate = 16000
        self.frame_duration = 30  # ms
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)
        
    async def start_continuous_recognition(self, callback: Callable[[str, float], None]):
        """Start continuous speech recognition."""
        self.is_recording = True
        
        # Start async recording
        recording_task = asyncio.create_task(self._record_audio())
        
        # Start VAD + transcription processing
        processing_task = asyncio.create_task(
            self._process_audio_stream(callback)
        )
        
        try:
            await asyncio.gather(recording_task, processing_task)
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
        finally:
            self.is_recording = False
    
    async def _record_audio(self):
        """Async audio recording."""
        def audio_callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio input status: {status}")
            
            # Add audio data to queue
            audio_data = indata.flatten().astype(np.float32)
            self.audio_queue.append(audio_data)
        
        # Start async recording with sounddevice
        with sd.InputStream(
            callback=audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.frame_size,
            dtype=np.float32
        ):
            while self.is_recording:
                await asyncio.sleep(0.1)
                
    async def _process_audio_stream(self, callback: Callable[[str, float], None]):
        """Process audio stream for speech detection and transcription."""
        speech_buffer = []
        silence_count = 0
        max_silence_frames = 10  # 300ms silence ends speech
        
        while self.is_recording:
            if not self.audio_queue:
                await asyncio.sleep(0.01)
                continue
                
            # Get frame
            frame = self.audio_queue.popleft()
            
            # VAD for speech detection
            frame_bytes = (frame * 32767).astype(np.int16).tobytes()
            is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
            
            if is_speech:
                speech_buffer.append(frame)
                silence_count = 0
            else:
                silence_count += 1
                
            # Transcribe when speech ends
            if silence_count >= max_silence_frames and speech_buffer:
                # Combine audio data
                audio_data = np.concatenate(speech_buffer)
                
                # Transcribe in separate task (avoid blocking)
                asyncio.create_task(
                    self._transcribe_audio(audio_data, callback)
                )
                
                speech_buffer = []
                silence_count = 0
                
    async def _transcribe_audio(self, audio_data: np.ndarray, callback: Callable[[str, float], None]):
        """Transcribe audio asynchronously."""
        try:
            # Run in executor to avoid blocking main thread
            loop = asyncio.get_event_loop()
            
            segments, info = await loop.run_in_executor(
                None,
                lambda: self.whisper_model.transcribe(
                    audio_data,
                    language="ja",
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=500,
                        speech_pad_ms=400
                    )
                )
            )
            
            # Combine transcription results
            transcription = " ".join([segment.text for segment in segments])
            
            if transcription.strip():
                await callback(transcription, info.language_probability)
                
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            
    async def transcribe_file(self, file_path: str) -> str:
        """Transcribe audio file."""
        loop = asyncio.get_event_loop()
        
        segments, info = await loop.run_in_executor(
            None,
            lambda: self.whisper_model.transcribe(
                file_path,
                language="ja",
                vad_filter=True
            )
        )
        
        return " ".join([segment.text for segment in segments])
        
    def stop_recording(self):
        """Stop continuous recording."""
        self.is_recording = False