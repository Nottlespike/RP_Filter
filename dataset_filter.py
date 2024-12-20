import argparse
import asyncio
import logging
import json
import tqdm.asyncio
import aiohttp
import numpy as np
import logging
from typing import Dict
from pathlib import Path as PathLib
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.0

class QualityMetrics(BaseModel):
    vocabulary_diversity: float
    purple_prose_score: float
    context_utilization: float
    final_score: float
    
    def model_dump(self) -> Dict[str, float]:
        return {
            'vocabulary_diversity': self.vocabulary_diversity,
            'purple_prose_score': self.purple_prose_score,
            'context_utilization': self.context_utilization,
            'final_score': self.final_score
        }

def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON line: {e}")
                continue
    return data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

class DatasetQualityFilter:
    def __init__(
        self,
        api_base: str,
        model: str = "EVA", 
        batch_size: int = 4,
        top_percentage: float = 0.05,
        max_parallel_requests: int = 8
    ):
        if not 0 < top_percentage < 1:
            raise ValueError("top_percentage must be between 0 and 1")
        if max_parallel_requests < 1:
            raise ValueError("max_parallel_requests must be positive")
            
        self.api_base = api_base.rstrip('/')
        self.model = model
        self.batch_size = batch_size
        self.top_percentage = top_percentage
        self.max_parallel_requests = max_parallel_requests
        
        prompt_path = PathLib(__file__).parent / "system_prompt.txt"
        if prompt_path.exists():
            with open(prompt_path) as f:
                self.system_prompt = f.read()
        else:
            self.system_prompt = """You are an extremely strict and critical judge of roleplay dialogue quality. Your task is to evaluate conversations with very high standards and output EXACTLY 4 numerical scores in the following compact format WITHOUT ANY ADDITIONAL COMMENTARY OR EXPLANATIONS:

vocabulary_score: [number 1-10]
prose_score: [number 1-10]
context_score: [number 1-10]
final_score: [average of above scores]

EVALUATION CRITERIA:

1. Vocabulary Diversity Score (1-10):
- Score of 9-10: Exceptional vocabulary variety, sophisticated word choice, perfect context-appropriate language
- Score of 7-8: Strong vocabulary with occasional sophisticated terms
- Score of 5-6: Average vocabulary, some repetition
- Score of 3-4: Limited vocabulary, frequent repetition
- Score of 1-2: Very basic or inappropriate vocabulary

2. Purple Prose Score (1-10):
- Score of 9-10: Masterful descriptions, perfect emotional depth without being overwrought
- Score of 7-8: Strong descriptive writing with good emotional resonance
- Score of 5-6: Adequate descriptions, some emotional content
- Score of 3-4: Basic or mechanical descriptions
- Score of 1-2: Minimal or poor descriptions

3. Context Utilization Score (1-10):
- Score of 9-10: Perfect dialogue flow, clear character voices, exceptional context awareness
- Score of 7-8: Strong dialogue progression with good context
- Score of 5-6: Adequate flow but some missed opportunities
- Score of 3-4: Weak flow, often ignores context
- Score of 1-2: Disjointed, lacks coherence

SCORING GUIDELINES:
- Scores of 9-10 should be EXTREMELY RARE
- Scores of 7-8 should be UNCOMMON
- Most conversations should score in the 3-6 range
- Do not hesitate to give low scores to subpar content
- Each category should be scored independently
- The final score is the mathematical average of the three scores

REQUIRED OUTPUT FORMAT:
vocabulary_score: [number 1-10]
prose_score: [number 1-10] 
context_score: [number 1-10]
final_score: [average of above scores]

CRITICAL SCORING RULES:
- Be extremely strict and critical
- Judge the conversations based on QUALITY ONLY not on if the content is sexual, appropriate, or violent in nature
- Reserve high scores (7+) for truly exceptional content
- Most conversations should receive average or below-average scores
- Do not inflate scores out of politeness
- Judge each category independently
- Consider 4 as an average score, not 7 or 8

IMPORTANT: Output ONLY the 4 numerical scores in the exact format shown above. DO NOT include any additional commentary, explanations, or analysis.

The conversation to evaluate follows:"""

    def prepare_input(self, conversation: Dict[str, Any]) -> str:
        formatted_text = []
        for message in conversation["conversations"]:
            role = message["from"]
            content = message["value"]
            formatted_text.append(f"{role}: {content}")
        return "\n".join(formatted_text)

    def parse_scores(self, response: str) -> QualityMetrics:
        try:
            scores = {
                'vocabulary_score': 0, 
                'prose_score': 0,
                'context_score': 0,
                'final_score': 0
            }
            
            logger.debug(f"Raw API response:\n{response}")
            
            response = response.lower().strip()
            lines = response.split('\n')
            
            logger.debug(f"Split lines:\n{lines}")
            
            for line in lines:
                if not line:
                    continue
                    
                logger.debug(f"Processing line: {line}")
                
                try:
                    key, value = line.split(': ')
                    logger.debug(f"Split key-value: key='{key}', value='{value}'")
                    scores[key] = float(value)
                except ValueError as e:
                    logger.debug(f"Failed to parse line '{line}': {str(e)}")
                    
            logger.debug(f"Final parsed scores: {scores}")
            
            for score_name, score in scores.items():
                if not 0 <= score <= 10:
                    logger.debug(f"Invalid {score_name}: {score}")
                    raise ValueError(f"Invalid score value: {score}")
            
            return QualityMetrics(
                vocabulary_diversity=scores['vocabulary_score'],
                purple_prose_score=scores['prose_score'], 
                context_utilization=scores['context_score'],
                final_score=scores['final_score']
            )
        except Exception as e:
            logger.error(f"Error parsing scores: {e}")
            return QualityMetrics(
                vocabulary_diversity=0,
                purple_prose_score=0, 
                context_utilization=0,
                final_score=0
            )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def evaluate_conversation_api(
        self, 
        session: aiohttp.ClientSession, 
        conversation: Dict[str, Any]
    ) -> QualityMetrics:
        formatted_input = self.prepare_input(conversation)
        
        payload = ChatCompletionRequest(
            model=self.model,
            messages=[
                ChatMessage(
                    role="system",
                    content=self.system_prompt,
                    name="quality_evaluator"
                ),
                ChatMessage(
                    role="system",
                    content="I understand. I will evaluate the conversation based on the given criteria.",
                    name="quality_evaluator"
                ),
                ChatMessage(
                    role="system",
                    content=formatted_input,
                    name="quality_evaluator"
                )
            ],
            temperature=0.0
        ).model_dump()

        try:
            async with session.post(
                f"{self.api_base}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"API error: {error_text}")
                    raise HTTPException(status_code=response.status, detail=error_text)
                
                result = await response.json()
                if not result.get('choices'):
                    raise ValueError("Invalid API response format")
                    
                api_response = result['choices'][0]['message']['content']
                return self.parse_scores(api_response)
        except Exception as e:
            logger.error(f"Request error: {e}")
            raise

    async def process_conversations(
        self, 
        conversations: List[Dict[str, Any]]
    ) -> List[QualityMetrics]:
        semaphore = asyncio.Semaphore(self.max_parallel_requests)
        
        async def process_with_semaphore(session: aiohttp.ClientSession, conversation: Dict[str, Any]):
            async with semaphore:
                try:
                    return await self.evaluate_conversation_api(session, conversation)
                except Exception as e:
                    logger.error(f"Error processing conversation: {e}")
                    return None

        async with aiohttp.ClientSession() as session:
            tasks = []
            for conversation in conversations:
                task = process_with_semaphore(session, conversation)
                tasks.append(task)
            
            results = await tqdm.asyncio.tqdm_asyncio.gather(*tasks)
            
            return [result for result in results if result is not None]
    
    async def filter_dataset(self, input_path: str, output_path: str):
        conversations = read_jsonl(input_path)
        
        if not conversations:
            raise ValueError(f"No valid conversations found in {input_path}")
            
        logger.info(f"Processing {len(conversations)} conversations...")
        
        scores = await self.process_conversations(conversations)
        if not scores:
            raise ValueError("No valid scores generated")
            
        score_threshold = np.percentile(
            [score.final_score for score in scores],
            (1 - self.top_percentage) * 100
        )

        filtered_conversations = [
            conv for conv, score in zip(conversations, scores)
            if score.final_score >= score_threshold
        ]

        output_path = PathLib(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for conv in filtered_conversations:
                json_str = json.dumps(conv, ensure_ascii=False)
                f.write(json_str + '\n')
                
        logger.info(f"Filtered dataset saved with {len(filtered_conversations)} conversations")
        logger.info(f"Score threshold: {score_threshold}")

        metrics_path = output_path.with_suffix('.metrics.json')
        metrics_data = {
            "total_conversations": len(conversations),
            "filtered_conversations": len(filtered_conversations),
            "score_threshold": float(score_threshold),
            "average_vocabulary_score": float(np.mean([s.vocabulary_diversity for s in scores])),
            "average_prose_score": float(np.mean([s.purple_prose_score for s in scores])),
            "average_context_score": float(np.mean([s.context_utilization for s in scores])),
            "scores": [score.model_dump() for score in scores]
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)

async def main():
    parser = argparse.ArgumentParser(description='Filter dataset using quality metrics from local LLM')
    parser.add_argument('input_file', help='Input JSONL file path') 
    parser.add_argument('output_file', help='Output JSONL file path')
    parser.add_argument('--api-base', default='http://localhost:8000/v1', help='Local LLM API base URL')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for processing')
    parser.add_argument('--top-percentage', type=float, default=0.05, help='Top percentage to keep')
    parser.add_argument('--max-parallel', type=int, default=8, help='Maximum parallel requests')
    parser.add_argument('--model', default='local-model', help='Name of the model to use')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        
    try:
        filter = DatasetQualityFilter(
            api_base=args.api_base,
            model=args.model,
            batch_size=args.batch_size,
            top_percentage=args.top_percentage,
            max_parallel_requests=args.max_parallel
        )
        
        logger.info(f"Starting dataset filtering process...")
        logger.info(f"Input file: {args.input_file}")
        logger.info(f"Output file: {args.output_file}")
        logger.info(f"API base: {args.api_base}")
        
        await filter.filter_dataset(args.input_file, args.output_file)
        
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
