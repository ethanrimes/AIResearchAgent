import asyncio
import time
import json
import psutil
import numpy as np
import aiohttp
import GPUtil
import argparse
from typing import List, Dict, Any

# CLI arguments
parser = argparse.ArgumentParser(description='Test Ollama inference performance with different concurrency levels')
parser.add_argument('--model', type=str, default='qwen3:0.6b', help='Model to test')
parser.add_argument('--prompt', type=str, default='Write me a lengthy story about a pig becoming elected president of the USA.', 
                    help='Prompt to use for testing')
parser.add_argument('--max_tokens', type=int, default=16000, help='Maximum context length')
parser.add_argument('--ollama_url', type=str, default='http://localhost:11434', help='Ollama API URL')
parser.add_argument('--concurrency_levels', type=str, default='1,2,4,6,8,12,16', 
                    help='Comma-separated list of concurrency levels to test')
args = parser.parse_args()

# Parse concurrency levels
concurrency_levels = [int(level) for level in args.concurrency_levels.split(',')]

# Monitoring class
class SystemMonitor:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.gpu_usages = []
        self.swap_usages = []
        self.running = False
        self.monitor_task = None
    
    async def start(self):
        self.running = True
        self.gpu_usages = []
        self.swap_usages = []
        self.monitor_task = asyncio.create_task(self._monitor())
    
    async def stop(self):
        if self.running:
            self.running = False
            if self.monitor_task:
                await self.monitor_task
    
    async def _monitor(self):
        while self.running:
            # GPU monitoring
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    self.gpu_usages.append(gpus[0].load * 100)  # Convert to percentage
                else:
                    self.gpu_usages.append(0)
            except:
                self.gpu_usages.append(0)
            
            # Swap monitoring
            swap = psutil.swap_memory()
            self.swap_usages.append(swap.percent)
            
            await asyncio.sleep(self.interval)
    
    def get_stats(self):
        if not self.gpu_usages:
            return {"gpu_avg": 0, "gpu_max": 0, "swap_avg": 0, "swap_max": 0}
        
        return {
            "gpu_avg": np.mean(self.gpu_usages),
            "gpu_max": np.max(self.gpu_usages),
            "swap_avg": np.mean(self.swap_usages),
            "swap_max": np.max(self.swap_usages)
        }

# Inference functions
async def generate_text(session, prompt, model, max_tokens):
    start_time = time.time()
    url = f"{args.ollama_url}/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens
        }
    }
    
    async with session.post(url, json=payload) as response:
        if response.status != 200:
            print(f"Error: {response.status}, {await response.text()}")
            return None
        
        result = await response.json()
        end_time = time.time()
        
        # Extract response and token information
        response_text = result.get("response", "")
        eval_count = result.get("eval_count", 0)  # Number of tokens generated
        eval_duration = result.get("eval_duration", 0) / 1e9  # Convert nanoseconds to seconds
        
        tokens_per_second = eval_count / eval_duration if eval_duration > 0 else 0
        
        return {
            "tokens": eval_count,
            "duration": eval_duration,
            "tokens_per_second": tokens_per_second,
            "total_time": end_time - start_time,
            "response": response_text[:100] + "..." if len(response_text) > 100 else response_text
        }

async def run_concurrent_requests(concurrency, model, prompt, max_tokens):
    monitor = SystemMonitor()
    await monitor.start()
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in range(concurrency):
            task = generate_text(session, prompt, model, max_tokens)
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
    
    await monitor.stop()
    system_stats = monitor.get_stats()
    
    # Filter out any failed requests
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print(f"All requests failed for concurrency {concurrency}")
        return None
    
    # Calculate statistics
    total_tokens = sum(r["tokens"] for r in valid_results)
    total_duration = end_time - start_time
    avg_tokens_per_second_per_request = np.mean([r["tokens_per_second"] for r in valid_results])
    total_tokens_per_second = total_tokens / total_duration
    
    return {
        "concurrency": concurrency,
        "total_tokens": total_tokens,
        "total_duration": total_duration,
        "total_tokens_per_second": total_tokens_per_second,
        "avg_tokens_per_second_per_request": avg_tokens_per_second_per_request,
        "gpu_usage_avg": system_stats["gpu_avg"],
        "gpu_usage_max": system_stats["gpu_max"],
        "swap_usage_avg": system_stats["swap_avg"],
        "swap_usage_max": system_stats["swap_max"]
    }

async def main():
    model = args.model
    prompt = args.prompt
    max_tokens = args.max_tokens
    
    print(f"Testing model: {model}")
    print(f"Max tokens: {max_tokens}")
    print(f"Concurrency levels: {concurrency_levels}")
    print("-" * 80)
    
    all_results = []
    
    for concurrency in concurrency_levels:
        print(f"Testing with {concurrency} concurrent requests...")
        result = await run_concurrent_requests(concurrency, model, prompt, max_tokens)
        
        if result:
            all_results.append(result)
            print(f"  Concurrency: {concurrency}")
            print(f"  Total tokens per second: {result['total_tokens_per_second']:.2f}")
            print(f"  Avg tokens per second per request: {result['avg_tokens_per_second_per_request']:.2f}")
            print(f"  GPU usage: {result['gpu_usage_avg']:.2f}% (avg), {result['gpu_usage_max']:.2f}% (max)")
            print(f"  Swap usage: {result['swap_usage_avg']:.2f}% (avg), {result['swap_usage_max']:.2f}% (max)")
            print("-" * 80)
    
    # Print summary table
    if all_results:
        print("\nSUMMARY:")
        print("-" * 80)
        print(f"{'Concurrency':<12} {'Tokens/s':<12} {'GPU Avg %':<12} {'GPU Max %':<12} {'Swap Avg %':<12} {'Swap Max %':<12}")
        print("-" * 80)
        
        for result in all_results:
            print(f"{result['concurrency']:<12} {result['total_tokens_per_second']:.2f}        {result['gpu_usage_avg']:.2f}         {result['gpu_usage_max']:.2f}         {result['swap_usage_avg']:.2f}          {result['swap_usage_max']:.2f}")
        
        # Find optimal concurrency
        optimal = max(all_results, key=lambda x: x['total_tokens_per_second'])
        print("-" * 80)
        print(f"Optimal concurrency: {optimal['concurrency']} with {optimal['total_tokens_per_second']:.2f} tokens/second")

if __name__ == "__main__":
    asyncio.run(main())