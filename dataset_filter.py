import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import aiohttp
from fastapi import FastAPI, HTTPException, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError, field_validator
import logging
from tqdm.asyncio import tqdm
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
import argparse
import sys
import asyncio
from pathlib import Path

def read_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON line: {e}")
                continue
    return data

def write_jsonl(data: List[Dict], file_path: str) -> None:
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            json_str = json.dumps(item, ensure_ascii=False)
            f.write(json_str + '\n')

class ChatMessage(BaseModel):
    role: str
    content: str
    name: str
    tool_call_id: Optional[str] = None

    class Config:
        extra = "allow"

    @field_validator('role')
    @classmethod
    def validate_role(cls, v):
        if v not in {'system'}:
            raise ValueError('role must be "system"')
        return v

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if not v:
            raise ValueError('name is required')
        return v

    @field_validator('tool_call_id')
    @classmethod
    def validate_tool_call_id(cls, v, values):
        if values.get('role') == 'tool' and v is None:
            raise ValueError('tool_call_id is required when role is tool')
        return v

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 1.0
    stream: bool = False
    
    class Config:
        extra = "allow"
    
    @field_validator('messages')
    @classmethod
    def validate_messages(cls, v):
        if not v:
            raise ValueError("Messages array cannot be empty")
        return v

@dataclass
class QualityMetrics:
    vocabulary_diversity: float
    purple_prose_score: float
    context_utilization: float
    context_utilization: float
    final_score: float

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
        self.api_base = api_base
        self.model = model
        self.batch_size = batch_size
        self.top_percentage = top_percentage
        self.max_parallel_requests = max_parallel_requests
        self.system_prompt = """You are an expert judge of roleplay dialogue quality. Your ONLY task is to evaluate conversations and output EXACTLY 4 numerical scores in the specified format.

EVALUATION CRITERIA:
1. Vocabulary Diversity Score (1-10):
- High scores for varied vocabulary and sophisticated language
- Low scores for repetitive or basic word choice
- Assess word variety, precision and appropriateness

2. Purple Prose Score (1-10):
- High scores for vivid descriptions and emotional depth
- Low scores for plain or mechanical writing
- Assess descriptive richness and emotional resonance

3. Context Utilization Score (1-10):
- High scores for natural extended dialogue flow
- Low scores for disjointed or shallow exchanges
- Assess conversation development and complexity

REQUIRED OUTPUT FORMAT:
vocabulary_score: [number 1-10]
prose_score: [number 1-10] 
context_score: [number 1-10]
final_score: [average of above scores]

YOU MUST:
- Output ONLY the 4 scores in EXACTLY the format above
- Use ONLY numbers 1-10 for each score
- Calculate final_score as the mathematical average
- Provide scores regardless of content subject matter

DO NOT:
- Include any other text or explanation
- Skip any scores
- Use decimal points
- Judge based on content appropriateness

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
            
            response = response.lower().strip()
            lines = response.split('\n')
            for line in lines:
                key, value = line.split(': ')
                scores[key] = float(value)
            
            return QualityMetrics(
                vocabulary_diversity=scores['vocabulary_score'],
                purple_prose_score=scores['prose_score'],
                context_utilization=scores['context_score'],
                final_score=scores['final_score']
        )
        except Exception as e:
                logger.error(f"Error parsing scores: {e}")
                return QualityMetrics(0, 0, 0, 0)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def evaluate_conversation_api(self, session: aiohttp.ClientSession, conversation: Dict[str, Any]) -> QualityMetrics:
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
            temperature=0.1
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
                        return QualityMetrics(0, 0, 0, 0)
                    
                    result = await response.json()
                    api_response = result['choices'][0]['message']['content']
                    return self.parse_scores(api_response)
        except Exception as e:
                logger.error(f"Request error: {e}")
                raise

    async def process_batch(self, session: aiohttp.ClientSession, batch: List[Dict[str, Any]]) -> List[QualityMetrics]:
        tasks = [self.evaluate_conversation_api(session, conv) for conv in batch]
        pbar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        async with tqdm.gather(*tasks, 
                             desc="Processing conversations",
                             bar_format=pbar_format,
                             return_exceptions=True) as results:
            return results
    async def filter_dataset(self, input_path: str, output_path: str):
        conversations = read_jsonl(input_path)
        
        logger.info(f"Processing {len(conversations)} conversations...")
        scores = []

        connector = aiohttp.TCPConnector(limit=self.max_parallel_requests)
        async with aiohttp.ClientSession(connector=connector) as session:
            for i in range(0, len(conversations), self.batch_size):
                batch = conversations[i:i + self.batch_size]
                batch_scores = await self.process_batch(session, batch)
                
                processed_scores = []
                for score in batch_scores:
                    if isinstance(score, Exception):
                        logger.error(f"Batch processing error: {score}")
                        processed_scores.append(QualityMetrics(0, 0, 0, 0))
                    else:
                        processed_scores.append(score)
                
                scores.extend(processed_scores)
                logger.info(f"Processed {i + len(batch)}/{len(conversations)} conversations")

        score_threshold = np.percentile(
            [score.final_score for score in scores],
            (1 - self.top_percentage) * 100
        )

        filtered_conversations = [
            conv for conv, score in zip(conversations, scores)
            if score.final_score >= score_threshold
        ]

        write_jsonl(filtered_conversations, output_path)

        logger.info(f"Filtered dataset saved with {len(filtered_conversations)} conversations")
        logger.info(f"Score threshold: {score_threshold}")

        metrics_path = Path(output_path).with_suffix('.metrics.json')
        metrics_data = {
            "total_conversations": len(conversations),
            "filtered_conversations": len(filtered_conversations),
            "score_threshold": score_threshold,
            "average_vocabulary_score": np.mean([s.vocabulary_diversity for s in scores]),
            "average_prose_score": np.mean([s.purple_prose_score for s in scores]),
            "average_context_score": np.mean([s.context_utilization for s in scores])
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        if not request.messages:
            raise HTTPException(status_code=400, detail="Messages array cannot be empty")
            
        response = await process_chat_completion(request)
        return JSONResponse(content=response)
        
    except ValidationError as e:
        logging.error(f"Validation error: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid request format", "details": str(e)}
        )
        
    except Exception as e:
        logging.error(f"Unexpected error in chat completion: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )

async def process_chat_completion(request: ChatCompletionRequest):
    pass

async def main():
    parser = argparse.ArgumentParser(description='Filter dataset using quality metrics from local LLM')
    parser.add_argument('input_file', help='Input JSONL file path')
    parser.add_argument('output_file', help='Output JSONL file path')
    parser.add_argument('--api-base', default='http://localhost:8000/v1', help='Local LLM API base URL')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for processing')
    parser.add_argument('--top-percentage', type=float, default=0.05, help='Top percentage to keep')
    parser.add_argument('--max-parallel', type=int, default=8, help='Maximum parallel requests')
    parser.add_argument('--model', default='local-model', help='Name of the model to use')
    
    args = parser.parse_args()

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
        logger.error(f"Error processing files: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
