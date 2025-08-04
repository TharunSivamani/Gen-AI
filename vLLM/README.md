```
docker run --runtime nvidia --gpus all -v {cwd}:/model --env "hf_api_key" -p 8000:8000 --ipc=host vllm/vllm-openai:latest --served-model-name Hammer2/Hammer2.0-0.5b --model /model --enforce-eager --gpu-memory-utilization 0.7
```

```
curl http://localhost:8000/v1/completions ^
  -H "Content-Type: application/json" ^
  -d "{
    \"model\": \"Hammer2/Hammer2.0-0.5b\",
    \"prompt\": \"Tell me a short story about a robot.\",
    \"max_tokens\": 100,
    \"temperature\": 0.7
  }"
```