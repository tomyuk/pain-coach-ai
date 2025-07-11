# Pain Coach AI Pascal 詳細設計書

## 1. システム概要とアーキテクチャ

### 1.1 全体アーキテクチャ（モジュラーモノリス）

```
┌─────────────────────────────────────────────────────────────┐
│                    MacBook Pro M3 Max                      │
├─────────────────────────────────────────────────────────────┤
│  Frontend Layer (macOS Native / Web Hybrid)                │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐│
│  │   Tauri App     │ │   Web Interface │ │  Voice UI       ││
│  │   (Rust/TS)     │ │   (Vue.js)      │ │  (WebRTC)       ││
│  └─────────────────┘ └─────────────────┘ └─────────────────┘│
├─────────────────────────────────────────────────────────────┤
│  API Gateway & Middleware Layer                            │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐│
│  │   FastAPI       │ │   WebSocket     │ │  Auth Manager   ││
│  │   (REST APIs)   │ │   (Real-time)   │ │  (KeyChain)     ││
│  └─────────────────┘ └─────────────────┘ └─────────────────┘│
├─────────────────────────────────────────────────────────────┤
│  Core Business Logic Modules                               │
│  ┌─────────────────┬─────────────────┬─────────────────────┐│
│  │  Pain Manager   │  AI Engine      │  Analytics Engine   ││
│  │  - Record pain  │  - MLX Runtime  │  - Pattern Analysis ││
│  │  - Track trends │  - LoRA Models  │  - Prediction ML    ││
│  │  - Symptoms     │  - Chat Logic   │  - Weather Corr.    ││
│  └─────────────────┼─────────────────┼─────────────────────┤│
│  │  Data Collector │  Privacy Guard  │  Health Connector   ││
│  │  - Sensor APIs  │  - Encryption   │  - HealthKit        ││
│  │  - Manual Input │  - GDPR Tools   │  - Fitbit API       ││
│  │  - Voice Input  │  - Audit Log    │  - Weather API      ││
│  └─────────────────┴─────────────────┴─────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│  Data Layer                                                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐│
│  │   DuckDB        │ │   Encrypted     │ │   Cache Layer   ││
│  │   (Analytics)   │ │   SQLite        │ │   (Redis)       ││
│  │                 │ │   (User Data)   │ │                 ││
│  └─────────────────┘ └─────────────────┘ └─────────────────┘│
├─────────────────────────────────────────────────────────────┤
│  AI/ML Layer (M3 Max Optimized)                           │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐│
│  │   MLX Runtime   │ │   Whisper STT   │ │   TTS Engine    ││
│  │   - Llama3.2    │ │   (faster-      │ │   (Style-BERT-  ││
│  │   - ELYZA-JP    │ │    whisper)     │ │    VITS2)       ││
│  │   - Custom LoRA │ │                 │ │                 ││
│  └─────────────────┘ └─────────────────┘ └─────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### 1.2 技術スタック選定理由

| コンポーネント | 選定技術 | 理由 |
|---|---|---|
| **フロントエンド** | Tauri + Vue.js | ネイティブ性能 + Web技術の柔軟性 |
| **バックエンド** | FastAPI | 高性能・型安全・MLライブラリ統合 |
| **AI エンジン** | MLX + llama.cpp | M3 Max最適化・将来性 |
| **データベース** | DuckDB + SQLite | 分析性能 + 軽量性 |
| **音声処理** | faster-whisper + Style-BERT-VITS2 | 日本語最適化・ローカル実行 |

## 2. データモデル設計

### 2.1 コアエンティティ設計

```sql
-- 患者プロファイル
CREATE TABLE users (
    id UUID PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- 基本情報（暗号化）
    name_encrypted BLOB,
    birth_year INTEGER,
    gender TEXT CHECK (gender IN ('male', 'female', 'other', 'prefer_not_to_say')),
    
    -- 医療情報
    primary_condition TEXT, -- 'fibromyalgia', 'chronic_back_pain', 'arthritis', etc.
    diagnosis_date DATE,
    medications JSONB,
    
    -- 設定
    timezone TEXT DEFAULT 'Asia/Tokyo',
    language TEXT DEFAULT 'ja',
    privacy_level INTEGER DEFAULT 2, -- 1: minimal, 2: standard, 3: comprehensive
    
    -- メタデータ
    data_version INTEGER DEFAULT 1,
    consent_version INTEGER DEFAULT 1,
    last_backup TIMESTAMP
);

-- 疼痛記録（最重要エンティティ）
CREATE TABLE pain_records (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- 疼痛評価
    pain_level INTEGER CHECK (pain_level >= 0 AND pain_level <= 10),
    pain_type TEXT[], -- ['sharp', 'dull', 'burning', 'aching', 'shooting']
    pain_locations JSONB, -- Body map coordinates
    pain_intensity_curve JSONB, -- Time-series within day
    
    -- コンテキスト
    weather_data JSONB,
    activity_before TEXT,
    sleep_quality INTEGER CHECK (sleep_quality >= 1 AND sleep_quality <= 5),
    stress_level INTEGER CHECK (stress_level >= 1 AND stress_level <= 5),
    
    -- 対処法
    medications_taken JSONB,
    non_med_interventions TEXT[],
    effectiveness_rating INTEGER,
    
    -- メタデータ
    input_method TEXT, -- 'manual', 'voice', 'api'
    confidence_score REAL, -- AI confidence in data quality
    source_device TEXT
);

-- AI対話ログ
CREATE TABLE ai_conversations (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    session_id UUID,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- 対話内容（暗号化）
    user_message_encrypted BLOB,
    ai_response_encrypted BLOB,
    
    -- コンテキスト
    conversation_type TEXT, -- 'pain_check', 'counseling', 'analysis', 'emergency'
    mood_detected TEXT,
    pain_context JSONB,
    
    -- AI処理情報
    model_used TEXT,
    processing_time_ms INTEGER,
    confidence_score REAL,
    
    -- プライバシー
    retention_until DATE,
    anonymized BOOLEAN DEFAULT FALSE
);

-- 分析結果キャッシュ
CREATE TABLE analysis_cache (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    analysis_type TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    valid_until TIMESTAMP,
    
    -- 結果データ
    result_data JSONB,
    confidence_interval JSONB,
    
    -- 計算メタデータ
    data_points_count INTEGER,
    algorithm_version TEXT,
    computation_time_ms INTEGER
);
```

### 2.2 時系列データ最適化（DuckDB特化）

```sql
-- DuckDB用の時系列最適化テーブル
CREATE TABLE pain_timeseries AS (
    SELECT 
        user_id,
        DATE_TRUNC('hour', recorded_at) as hour_bucket,
        AVG(pain_level) as avg_pain,
        MIN(pain_level) as min_pain,
        MAX(pain_level) as max_pain,
        COUNT(*) as record_count,
        LIST(pain_type) as pain_types_agg,
        FIRST(weather_data) as weather_snapshot
    FROM pain_records 
    GROUP BY user_id, hour_bucket
);

-- パーティション戦略（月次）
CREATE TABLE pain_records_partitioned (
    LIKE pain_records INCLUDING ALL
) PARTITION BY RANGE (recorded_at);

-- インデックス戦略
CREATE INDEX idx_pain_user_time ON pain_records (user_id, recorded_at DESC);
CREATE INDEX idx_pain_level_location ON pain_records USING GIN (pain_locations);
```

## 3. AI エンジン詳細設計

### 3.1 MLX統合アーキテクチャ

```python
# core/ai/mlx_engine.py
from typing import Optional, Dict, List, AsyncGenerator
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
import asyncio
from dataclasses import dataclass

@dataclass
class PainChatConfig:
    model_name: str = "ELYZA-japanese-Llama-3-8B-Instruct"
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    context_window: int = 4096
    lora_path: Optional[str] = None

class PainCoachMLXEngine:
    def __init__(self, config: PainChatConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.lora_loaded = False
        
    async def initialize(self):
        """M3 Max最適化でモデル初期化"""
        # Metal Performance Shadersを活用
        mx.set_default_device(mx.gpu)
        
        # モデルとトークナイザーのロード
        self.model, self.tokenizer = load(
            self.config.model_name,
            adapter_path=self.config.lora_path
        )
        
        # ウォームアップ実行
        await self._warmup()
        
    async def _warmup(self):
        """推論エンジンウォームアップ"""
        test_prompt = "こんにちは、調子はいかがですか？"
        await self.generate_response(test_prompt, max_tokens=10)
        
    async def generate_response(
        self, 
        prompt: str, 
        pain_context: Optional[Dict] = None,
        max_tokens: int = None
    ) -> AsyncGenerator[str, None]:
        """ストリーミング応答生成"""
        
        # コンテキスト拡張プロンプト
        enhanced_prompt = self._build_context_prompt(prompt, pain_context)
        
        # 非同期ストリーミング生成
        async for token in self._async_generate(enhanced_prompt, max_tokens):
            yield token
            
    def _build_context_prompt(self, user_input: str, pain_context: Dict) -> str:
        """疼痛コンテキストを含むプロンプト構築"""
        system_prompt = """あなたは慢性疼痛患者を支援する専門的なAIコーチです。
共感的で、医学的に正確で、希望を与える対話を心がけてください。
決して医学的診断や処方を行わず、医療従事者への相談を推奨してください。"""
        
        context_info = ""
        if pain_context:
            context_info = f"""
現在の痛みレベル: {pain_context.get('current_pain', 'N/A')}/10
最近のパターン: {pain_context.get('recent_pattern', 'N/A')}
主な痛みの箇所: {pain_context.get('main_locations', 'N/A')}
"""
        
        return f"""<|system|>
{system_prompt}

{context_info}
<|user|>
{user_input}
<|assistant|>
"""

    async def _async_generate(self, prompt: str, max_tokens: int) -> AsyncGenerator[str, None]:
        """非同期トークン生成"""
        loop = asyncio.get_event_loop()
        
        # CPU集約的な処理を別スレッドで実行
        tokens = await loop.run_in_executor(
            None, 
            lambda: generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p
            )
        )
        
        # トークンをストリーミング
        for token in tokens:
            yield self.tokenizer.decode([token])
            await asyncio.sleep(0.01)  # UIの応答性確保
```

### 3.2 カスタムLoRA実装

```python
# core/ai/pain_lora.py
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from unsloth import FastLanguageModel
import pandas as pd

class PainSpecificLoRATrainer:
    def __init__(self, base_model: str = "ELYZA-japanese-Llama-3-8B"):
        self.base_model = base_model
        self.model = None
        self.tokenizer = None
        
    def prepare_training_data(self) -> Dataset:
        """慢性疼痛特化の訓練データ準備"""
        
        # 合成データ生成（差分プライバシー適用）
        pain_scenarios = [
            {
                "input": "今日は痛みが酷くて何もできません",
                "output": "辛い状況ですね。痛みが強い日は無理をせず、できる範囲で過ごすことが大切です。呼吸法や軽いストレッチなど、症状を和らげる方法を一緒に探してみませんか？"
            },
            {
                "input": "線維筋痛症と診断されて不安です",
                "output": "診断を受けることで不安になるのは自然な反応です。線維筋痛症は適切な管理により症状をコントロールできる疾患です。医師との連携を大切にしながら、日常生活での工夫を一緒に考えていきましょう。"
            },
            # ... 1000件以上の対話例
        ]
        
        # Unslothフォーマットに変換
        formatted_data = []
        for scenario in pain_scenarios:
            formatted_data.append({
                "conversations": [
                    {"from": "human", "value": scenario["input"]},
                    {"from": "gpt", "value": scenario["output"]}
                ]
            })
            
        return Dataset.from_pandas(pd.DataFrame(formatted_data))
    
    async def train_lora(self, epochs: int = 3, batch_size: int = 4):
        """QLoRA訓練実行"""
        
        # Unslothでモデル準備（M3 Max最適化）
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model,
            max_seq_length=2048,
            dtype=None,  # Auto detect
            load_in_4bit=True,  # QLoRA
        )
        
        # LoRA設定
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,  # Rank
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        
        # 訓練データ準備
        dataset = self.prepare_training_data()
        
        # 訓練実行
        from unsloth import UnslothTrainer, UnslothTrainingArguments
        
        trainer = UnslothTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="conversations",
            max_seq_length=2048,
            dataset_num_proc=2,
            
            args=UnslothTrainingArguments(
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=1,
                warmup_steps=5,
                num_train_epochs=epochs,
                learning_rate=2e-4,
                fp16=True,
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=42,
                output_dir="./pain_lora_output",
            ),
        )
        
        trainer_stats = trainer.train()
        
        # LoRA保存
        self.model.save_pretrained("./models/pain_lora")
        self.tokenizer.save_pretrained("./models/pain_lora")
        
        return trainer_stats
```

## 4. 音声処理パイプライン

### 4.1 リアルタイム音声認識

```python
# core/audio/speech_processor.py
import asyncio
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import webrtcvad
from collections import deque
import threading

class RealTimeSpeechProcessor:
    def __init__(self, model_size: str = "large-v3"):
        # Whisperモデル初期化（M3 Max最適化）
        self.whisper_model = WhisperModel(
            model_size, 
            device="auto",  # M3 Max自動検出
            compute_type="int8"  # 量子化で高速化
        )
        
        # VAD（Voice Activity Detection）
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level
        
        # オーディオバッファ
        self.audio_queue = deque(maxlen=30)  # 30秒バッファ
        self.is_recording = False
        
        # 設定
        self.sample_rate = 16000
        self.frame_duration = 30  # ms
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)
        
    async def start_continuous_recognition(self, callback):
        """連続音声認識開始"""
        self.is_recording = True
        
        # 非同期録音開始
        recording_task = asyncio.create_task(self._record_audio())
        
        # VAD + 転写処理
        processing_task = asyncio.create_task(
            self._process_audio_stream(callback)
        )
        
        await asyncio.gather(recording_task, processing_task)
    
    async def _record_audio(self):
        """非同期オーディオ録音"""
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio input status: {status}")
            
            # オーディオデータをキューに追加
            audio_data = indata.flatten().astype(np.float32)
            self.audio_queue.append(audio_data)
        
        # sounddeviceで非同期録音
        with sd.InputStream(
            callback=audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.frame_size,
            dtype=np.float32
        ):
            while self.is_recording:
                await asyncio.sleep(0.1)
                
    async def _process_audio_stream(self, callback):
        """音声ストリーム処理"""
        speech_buffer = []
        silence_count = 0
        max_silence_frames = 10  # 300ms無音で発話終了
        
        while self.is_recording:
            if not self.audio_queue:
                await asyncio.sleep(0.01)
                continue
                
            # フレーム取得
            frame = self.audio_queue.popleft()
            
            # VADで音声検出
            frame_bytes = (frame * 32767).astype(np.int16).tobytes()
            is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
            
            if is_speech:
                speech_buffer.append(frame)
                silence_count = 0
            else:
                silence_count += 1
                
            # 発話終了検出時に転写実行
            if silence_count >= max_silence_frames and speech_buffer:
                # 音声データ統合
                audio_data = np.concatenate(speech_buffer)
                
                # 別スレッドで転写実行（ブロッキング回避）
                asyncio.create_task(
                    self._transcribe_audio(audio_data, callback)
                )
                
                speech_buffer = []
                silence_count = 0
                
    async def _transcribe_audio(self, audio_data: np.ndarray, callback):
        """音声転写（非同期）"""
        try:
            # メインスレッドをブロックしないよう別スレッドで実行
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
            
            # 転写結果を統合
            transcription = " ".join([segment.text for segment in segments])
            
            if transcription.strip():
                await callback(transcription, info.language_probability)
                
        except Exception as e:
            print(f"転写エラー: {e}")
```

### 4.2 感情配慮型TTS

```python
# core/audio/tts_engine.py
import torch
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.tts_model import TTSModel
import soundfile as sf
import io
import asyncio

class EmpatheticTTSEngine:
    def __init__(self, model_path: str):
        self.model = TTSModel.from_pretrained(model_path)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(self.device)
        
    async def synthesize_empathetic_speech(
        self, 
        text: str, 
        emotion: str = "gentle",
        pain_context: dict = None
    ) -> bytes:
        """共感的音声合成"""
        
        # 感情・コンテキストに基づくパラメータ調整
        style_params = self._get_style_parameters(emotion, pain_context)
        
        # 非同期音声合成
        loop = asyncio.get_event_loop()
        audio_data = await loop.run_in_executor(
            None,
            self._synthesize,
            text,
            style_params
        )
        
        return audio_data
    
    def _get_style_parameters(self, emotion: str, pain_context: dict):
        """感情・痛みレベルに応じた音声スタイル調整"""
        base_params = {
            "speed": 1.0,
            "pitch": 0.0,
            "intonation": 1.0,
            "volume": 0.8
        }
        
        # 痛みレベルに応じた調整
        if pain_context and "pain_level" in pain_context:
            pain_level = pain_context["pain_level"]
            
            if pain_level >= 7:  # 高痛時
                base_params.update({
                    "speed": 0.9,  # ゆっくり
                    "pitch": -0.1,  # 少し低く
                    "intonation": 0.8,  # 穏やかに
                    "volume": 0.7  # 小さめに
                })
            elif pain_level <= 3:  # 低痛時
                base_params.update({
                    "speed": 1.1,  # やや速く
                    "intonation": 1.2,  # 明るく
                    "volume": 0.9
                })
        
        # 感情による調整
        emotion_adjustments = {
            "gentle": {"pitch": -0.05, "speed": 0.95},
            "encouraging": {"pitch": 0.05, "speed": 1.05, "intonation": 1.1},
            "calm": {"pitch": -0.1, "speed": 0.9, "intonation": 0.9},
            "supportive": {"pitch": 0.0, "speed": 1.0, "intonation": 1.05}
        }
        
        if emotion in emotion_adjustments:
            base_params.update(emotion_adjustments[emotion])
            
        return base_params
    
    def _synthesize(self, text: str, style_params: dict) -> bytes:
        """実際の音声合成処理"""
        with torch.no_grad():
            # Style-BERT-VITS2で音声生成
            audio = self.model.infer(
                text=text,
                speed=style_params["speed"],
                pitch=style_params["pitch"],
                intonation=style_params["intonation"],
                volume=style_params["volume"]
            )[0]
            
            # WAVデータとしてエンコード
            buffer = io.BytesIO()
            sf.write(buffer, audio, 22050, format='WAV')
            return buffer.getvalue()
```

## 5. データ統合・プライバシー設計

### 5.1 ヘルスケアAPI統合マネージャー

```python
# core/integrations/health_manager.py
from abc import ABC, abstractmethod
import aiohttp
import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json

class HealthDataProvider(ABC):
    @abstractmethod
    async def fetch_data(self, start_date: datetime, end_date: datetime) -> Dict:
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        pass

class FitbitProvider(HealthDataProvider):
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.base_url = "https://api.fitbit.com/1/user/-"
        
    async def fetch_data(self, start_date: datetime, end_date: datetime) -> Dict:
        """Fitbitデータ取得"""
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        async with aiohttp.ClientSession() as session:
            # 並列データ取得
            tasks = [
                self._fetch_heart_rate(session, headers, start_date, end_date),
                self._fetch_sleep_data(session, headers, start_date, end_date),
                self._fetch_activity_data(session, headers, start_date, end_date),
                self._fetch_stress_data(session, headers, start_date, end_date)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            return {
                "heart_rate": results[0] if not isinstance(results[0], Exception) else None,
                "sleep": results[1] if not isinstance(results[1], Exception) else None,
                "activity": results[2] if not isinstance(results[2], Exception) else None,
                "stress": results[3] if not isinstance(results[3], Exception) else None,
                "source": "fitbit",
                "timestamp": datetime.now()
            }
    
    async def _fetch_heart_rate(self, session, headers, start_date, end_date):
        """心拍数データ取得"""
        url = f"{self.base_url}/activities/heart/date/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}.json"
        
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("activities-heart", [])
            return None

class AppleHealthProvider(HealthDataProvider):
    """Apple HealthKit統合（Shortcuts経由）"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        
    async def fetch_data(self, start_date: datetime, end_date: datetime) -> Dict:
        """Apple Shortcuts経由でHealthKitデータ取得"""
        
        # Shortcutsアプリへの要求データ
        request_data = {
            "action": "fetch_health_data",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "data_types": [
                "step_count",
                "heart_rate",
                "sleep_analysis",
                "mindful_minutes"
            ]
        }
        
        async with aiohttp.ClientSession() as session:
            # Webhook経由でShortcutsにリクエスト
            async with session.post(
                self.webhook_url, 
                json=request_data,
                timeout=30
            ) as response:
                if response.status == 200:
                    return await response.json()
                return None

class WeatherProvider:
    """OpenWeatherMap統合"""
    
    def __init__(self, api_key: str, location: tuple):
        self.api_key = api_key
        self.lat, self.lon = location
        
    async def fetch_weather_history(self, days_back: int = 7) -> List[Dict]:
        """過去の気象データ取得"""
        weather_data = []
        
        async with aiohttp.ClientSession() as session:
            for i in range(days_back):
                target_date = datetime.now() - timedelta(days=i)
                timestamp = int(target_date.timestamp())
                
                url = f"https://api.openweathermap.org/data/2.5/onecall/timemachine"
                params = {
                    "lat": self.lat,
                    "lon": self.lon,
                    "dt": timestamp,
                    "appid": self.api_key,
                    "units": "metric"
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # 疼痛関連気象要素を抽出
                        weather_point = {
                            "date": target_date.date(),
                            "pressure": data["current"]["pressure"],
                            "humidity": data["current"]["humidity"],
                            "temperature": data["current"]["temp"],
                            "pressure_change": None,  # 後で計算
                            "weather_desc": data["current"]["weather"][0]["description"]
                        }
                        
                        weather_data.append(weather_point)
                        
                await asyncio.sleep(0.1)  # API制限考慮
        
        # 気圧変化を計算
        for i in range(1, len(weather_data)):
            pressure_change = weather_data[i-1]["pressure"] - weather_data[i]["pressure"]
            weather_data[i]["pressure_change"] = pressure_change
            
        return weather_data

class HealthDataManager:
    """統合ヘルスデータマネージャー"""
    
    def __init__(self):
        self.providers: Dict[str, HealthDataProvider] = {}
        self.weather_provider: Optional[WeatherProvider] = None
        
    def register_provider(self, name: str, provider: HealthDataProvider):
        """ヘルスデータプロバイダー登録"""
        self.providers[name] = provider
        
    def register_weather_provider(self, provider: WeatherProvider):
        """天気データプロバイダー登録"""
        self.weather_provider = provider
        
    async def collect_comprehensive_data(self, days_back: int = 7) -> Dict:
        """包括的データ収集"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # 並列データ収集
        tasks = []
        
        # ヘルスデータ
        for name, provider in self.providers.items():
            tasks.append(self._safe_fetch(name, provider, start_date, end_date))
            
        # 気象データ
        if self.weather_provider:
            tasks.append(self._fetch_weather_data(days_back))
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 結果統合
        integrated_data = {
            "collection_timestamp": datetime.now(),
            "date_range": {"start": start_date, "end": end_date},
            "health_data": {},
            "weather_data": None,
            "errors": []
        }
        
        # 結果処理
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                integrated_data["errors"].append(str(result))
            elif i < len(self.providers):
                provider_name = list(self.providers.keys())[i]
                integrated_data["health_data"][provider_name] = result
            else:
                integrated_data["weather_data"] = result
                
        return integrated_data
    
    async def _safe_fetch(self, name: str, provider: HealthDataProvider, start_date: datetime, end_date: datetime):
        """安全なデータ取得（エラーハンドリング付き）"""
        try:
            return await provider.fetch_data(start_date, end_date)
        except Exception as e:
            print(f"Provider {name} error: {e}")
            return None
            
    async def _fetch_weather_data(self, days_back: int):
        """気象データ取得"""
        try:
            return await self.weather_provider.fetch_weather_history(days_back)
        except Exception as e:
            print(f"Weather data error: {e}")
            return None
```

### 5.2 プライバシー保護システム

```python
# core/privacy/encryption_manager.py
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import keyring
import base64
import os
import json
from typing import Dict, Any, Optional
import hashlib

class EncryptionManager:
    """医療データ暗号化マネージャー"""
    
    def __init__(self, app_name: str = "PainCoachAI"):
        self.app_name = app_name
        self.salt = self._get_or_create_salt()
        self._cipher_cache: Optional[Fernet] = None
        
    def _get_or_create_salt(self) -> bytes:
        """ソルト取得または生成"""
        try:
            salt_b64 = keyring.get_password(self.app_name, "encryption_salt")
            if salt_b64:
                return base64.b64decode(salt_b64)
        except:
            pass
            
        # 新しいソルト生成
        salt = os.urandom(16)
        keyring.set_password(
            self.app_name, 
            "encryption_salt", 
            base64.b64encode(salt).decode()
        )
        return salt
    
    def _get_cipher(self, user_id: str) -> Fernet:
        """ユーザー固有の暗号化キー取得"""
        if self._cipher_cache:
            return self._cipher_cache
            
        try:
            # KeyChainからキー取得
            key_b64 = keyring.get_password(self.app_name, f"user_key_{user_id}")
            if key_b64:
                key = base64.b64decode(key_b64)
            else:
                # 新しいキー生成
                key = Fernet.generate_key()
                keyring.set_password(
                    self.app_name, 
                    f"user_key_{user_id}", 
                    base64.b64encode(key).decode()
                )
        except:
            # フォールバック: パスワードベースキー
            password = f"pain_coach_user_{user_id}"
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=self.salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            
        self._cipher_cache = Fernet(key)
        return self._cipher_cache
    
    def encrypt_data(self, data: Any, user_id: str) -> bytes:
        """データ暗号化"""
        cipher = self._get_cipher(user_id)
        
        # JSON文字列に変換
        json_str = json.dumps(data, ensure_ascii=False, default=str)
        
        # 暗号化
        encrypted = cipher.encrypt(json_str.encode('utf-8'))
        return encrypted
    
    def decrypt_data(self, encrypted_data: bytes, user_id: str) -> Any:
        """データ復号化"""
        cipher = self._get_cipher(user_id)
        
        try:
            # 復号化
            decrypted_bytes = cipher.decrypt(encrypted_data)
            json_str = decrypted_bytes.decode('utf-8')
            
            # JSON解析
            return json.loads(json_str)
        except Exception as e:
            raise ValueError(f"復号化失敗: {e}")
    
    def hash_sensitive_data(self, data: str) -> str:
        """センシティブデータのハッシュ化（検索用）"""
        return hashlib.sha256(
            (data + base64.b64encode(self.salt).decode()).encode()
        ).hexdigest()

class GDPRComplianceManager:
    """GDPR準拠管理"""
    
    def __init__(self, db_manager):
        self.db = db_manager
        
    async def process_data_deletion_request(self, user_id: str) -> Dict:
        """データ削除要求処理"""
        deletion_summary = {
            "user_id": user_id,
            "timestamp": datetime.now(),
            "deleted_records": {},
            "errors": []
        }
        
        try:
            # 全テーブルからユーザーデータ削除
            tables_to_clean = [
                "pain_records",
                "ai_conversations", 
                "analysis_cache",
                "user_preferences",
                "health_integrations"
            ]
            
            for table in tables_to_clean:
                count = await self.db.execute(
                    f"DELETE FROM {table} WHERE user_id = ?",
                    (user_id,)
                )
                deletion_summary["deleted_records"][table] = count
                
            # ユーザープロファイル削除
            await self.db.execute(
                "DELETE FROM users WHERE id = ?",
                (user_id,)
            )
            
            # キーチェーンからキー削除
            try:
                keyring.delete_password("PainCoachAI", f"user_key_{user_id}")
            except:
                pass
                
        except Exception as e:
            deletion_summary["errors"].append(str(e))
            
        return deletion_summary
    
    async def export_user_data(self, user_id: str) -> Dict:
        """ユーザーデータエクスポート（データポータビリティ）"""
        export_data = {
            "export_timestamp": datetime.now(),
            "user_id": user_id,
            "data": {}
        }
        
        # 全データ取得
        tables = [
            "users", "pain_records", "ai_conversations", 
            "analysis_cache", "user_preferences"
        ]
        
        for table in tables:
            rows = await self.db.fetch_all(
                f"SELECT * FROM {table} WHERE user_id = ? OR id = ?",
                (user_id, user_id)
            )
            
            # 暗号化データは復号化してエクスポート
            export_data["data"][table] = []
            for row in rows:
                row_dict = dict(row)
                
                # 暗号化フィールドの復号化
                for field, value in row_dict.items():
                    if field.endswith("_encrypted") and value:
                        try:
                            decrypted = self.encryption_manager.decrypt_data(value, user_id)
                            row_dict[field.replace("_encrypted", "")] = decrypted
                            del row_dict[field]
                        except:
                            pass
                            
                export_data["data"][table].append(row_dict)
                
        return export_data
```

## 6. パフォーマンス最適化と監視

### 6.1 M3 Max特化最適化

```python
# core/performance/m3_optimizer.py
import psutil
import torch
import mlx.core as mx
from concurrent.futures import ThreadPoolExecutor
import asyncio
from typing import Dict, Any
import time

class M3MaxOptimizer:
    """M3 Max特化パフォーマンス最適化"""
    
    def __init__(self):
        self.cpu_count = psutil.cpu_count()
        self.memory_total = psutil.virtual_memory().total
        self.gpu_available = torch.backends.mps.is_available()
        
        # MLX最適設定
        if mx.metal.is_available():
            mx.set_default_device(mx.gpu)
            
        # スレッドプール初期化
        self.thread_pool = ThreadPoolExecutor(
            max_workers=min(8, self.cpu_count)
        )
        
    async def optimize_ai_inference(self, model_config: Dict) -> Dict:
        """AI推論最適化"""
        optimization_settings = {
            "batch_size": 1,  # リアルタイム対話用
            "max_workers": 4,  # M3 Max効率CPU使用
            "memory_pool_size": "auto",
            "gpu_utilization": 0.8,  # GPU使用率上限
        }
        
        # メモリ使用量に基づく調整
        available_memory = psutil.virtual_memory().available
        if available_memory < 8 * 1024**3:  # 8GB未満
            optimization_settings.update({
                "quantization": "int8",
                "model_offload": True,
                "cache_size": "small"
            })
        elif available_memory > 16 * 1024**3:  # 16GB以上
            optimization_settings.update({
                "quantization": "int4",
                "model_offload": False,
                "cache_size": "large"
            })
            
        return optimization_settings
    
    async def monitor_system_resources(self) -> Dict[str, Any]:
        """システムリソース監視"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "temperature": self._get_cpu_temperature(),
            "gpu_utilization": self._get_gpu_utilization(),
            "disk_io": psutil.disk_io_counters()._asdict(),
            "network_io": psutil.net_io_counters()._asdict()
        }
    
    def _get_cpu_temperature(self) -> float:
        """CPU温度取得（macOS）"""
        try:
            # powermetrics経由で取得
            import subprocess
            result = subprocess.run(
                ["powermetrics", "-n", "1", "-s", "cpu_power"],
                capture_output=True, text=True, timeout=5
            )
            # 温度情報をパース（簡略版）
            return 0.0  # 実装省略
        except:
            return 0.0
    
    def _get_gpu_utilization(self) -> float:
        """GPU使用率取得"""
        if self.gpu_available:
            try:
                # Metal Performance Shaders使用率取得
                return mx.metal.get_active_memory() / mx.metal.get_peak_memory()
            except:
                return 0.0
        return 0.0

class PerformanceProfiler:
    """パフォーマンスプロファイラー"""
    
    def __init__(self):
        self.metrics = {}
        
    async def profile_ai_response_time(self, ai_engine, prompt: str) -> Dict:
        """AI応答時間プロファイリング"""
        start_time = time.perf_counter()
        
        # メモリ使用量測定開始
        memory_before = psutil.Process().memory_info().rss
        
        # AI推論実行
        response = await ai_engine.generate_response(prompt)
        
        # 完了時間測定
        end_time = time.perf_counter()
        memory_after = psutil.Process().memory_info().rss
        
        return {
            "response_time_ms": (end_time - start_time) * 1000,
            "memory_delta_mb": (memory_after - memory_before) / 1024 / 1024,
            "tokens_per_second": len(response.split()) / (end_time - start_time),
            "prompt_length": len(prompt),
            "response_length": len(response)
        }
    
    async def benchmark_database_operations(self, db_manager) -> Dict:
        """データベース操作ベンチマーク"""
        benchmarks = {}
        
        # 書き込み性能
        start_time = time.perf_counter()
        for i in range(100):
            await db_manager.insert_pain_record({
                "pain_level": 5,
                "timestamp": time.time(),
                "location": "test"
            })
        write_time = time.perf_counter() - start_time
        benchmarks["write_ops_per_sec"] = 100 / write_time
        
        # 読み込み性能
        start_time = time.perf_counter()
        for i in range(100):
            await db_manager.get_recent_pain_records(limit=10)
        read_time = time.perf_counter() - start_time
        benchmarks["read_ops_per_sec"] = 100 / read_time
        
        # 分析クエリ性能
        start_time = time.perf_counter()
        await db_manager.analyze_pain_trends(days=30)
        analysis_time = time.perf_counter() - start_time
        benchmarks["analysis_time_ms"] = analysis_time * 1000
        
        return benchmarks
```

この詳細設計書では、Pain Coach AI Pascalの実装に必要な全コンポーネントの技術仕様を明確化しました。特にM3 Max環境での最適化、医療データのプライバシー保護、リアルタイム音声処理に焦点を当てて設計しています。

次のステップとして、どの部分の実装を最初に進めるか、または特定のコンポーネントの詳細化が必要でしたらお知らせください。