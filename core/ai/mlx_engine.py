"""MLX-powered AI engine for Pain Coach AI Pascal."""

from typing import Optional, Dict, List, AsyncGenerator
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
import asyncio
from dataclasses import dataclass
import time

@dataclass
class PainChatConfig:
    """Configuration for Pain Coach AI MLX engine."""
    model_name: str = "ELYZA-japanese-Llama-3-8B-Instruct"
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    context_window: int = 4096
    lora_path: Optional[str] = None

class PainCoachMLXEngine:
    """Main AI engine powered by MLX for M3 Max optimization."""
    
    def __init__(self, config: PainChatConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.lora_loaded = False
        self.initialized = False
        
    async def initialize(self):
        """Initialize model with M3 Max optimization."""
        if self.initialized:
            return
            
        # Metal Performance Shaders optimization
        if mx.metal.is_available():
            mx.set_default_device(mx.gpu)
        
        # Load model and tokenizer
        self.model, self.tokenizer = load(
            self.config.model_name,
            adapter_path=self.config.lora_path
        )
        
        # Warmup inference
        await self._warmup()
        self.initialized = True
        
    async def _warmup(self):
        """Warmup inference engine."""
        test_prompt = "こんにちは、調子はいかがですか？"
        async for _ in self.generate_response(test_prompt, max_tokens=10):
            pass
        
    async def generate_response(
        self, 
        prompt: str, 
        pain_context: Optional[Dict] = None,
        max_tokens: int = None
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response."""
        if not self.initialized:
            await self.initialize()
            
        # Build context-enhanced prompt
        enhanced_prompt = self._build_context_prompt(prompt, pain_context)
        
        # Async streaming generation
        async for token in self._async_generate(enhanced_prompt, max_tokens):
            yield token
            
    def _build_context_prompt(self, user_input: str, pain_context: Optional[Dict]) -> str:
        """Build context-enhanced prompt for pain coaching."""
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
        """Async token generation."""
        loop = asyncio.get_event_loop()
        
        # Run CPU-intensive generation in thread executor
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
        
        # Stream tokens
        for token in tokens:
            yield self.tokenizer.decode([token])
            await asyncio.sleep(0.01)  # Ensure UI responsiveness
            
    async def analyze_pain_context(self, pain_records: List[Dict]) -> Dict:
        """Analyze pain patterns and provide insights."""
        # Simple analysis - could be enhanced with ML
        if not pain_records:
            return {"insights": [], "recommendations": []}
            
        levels = [r.get("pain_level", 0) for r in pain_records]
        avg_pain = sum(levels) / len(levels)
        
        insights = []
        recommendations = []
        
        if avg_pain > 7:
            insights.append("高い痛みレベルが継続しています")
            recommendations.append("医療従事者との相談を検討してください")
        elif avg_pain < 4:
            insights.append("痛みレベルは比較的安定しています")
            recommendations.append("現在の対処法を継続してください")
            
        return {
            "average_pain": avg_pain,
            "insights": insights,
            "recommendations": recommendations
        }