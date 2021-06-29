## Build docker image
### GPU image
```bash
wget https://data.dgl.ai/dataset/FB15k.zip -P install/
docker build -t dgl-gpu:torch-1.2.0-cu11 -f Dockerfile.ci_gpu_cu11 .
```
