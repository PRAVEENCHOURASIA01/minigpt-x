# Deployment Guide for MiniGPT-X

This guide provides copy-paste commands to deploy MiniGPT-X to Hugging Face Spaces and publish to GitHub.

---

## 1. Create and Deploy Hugging Face Space (Docker SDK)

1. Go to [https://huggingface.co/new-space](https://huggingface.co/new-space)
   - Space Name: `minigpt-x`
   - License: `mit`
   - Space SDK: **Docker** (Blank)
   - Visibility: Public

2. In your terminal, clone your newly created Space repository:
   ```bash
   git clone https://huggingface.co/spaces/<YOUR_USERNAME>/minigpt-x ~/minigpt-x-space
   ```

3. Export the desired model checkpoint (e.g. `main_best.pt` or `smoke_latest.pt`):
   ```bash
   cd /Users/praveenchourasia/Projects/minigpt-x
   /usr/local/bin/python3 -m scripts.export_checkpoint --input checkpoints/main_best.pt --output deploy/hf_space/model.pt
   ```

4. Copy the deployment files into your Space repository:
   ```bash
   cp -R deploy/hf_space/* ~/minigpt-x-space/
   cd ~/minigpt-x-space
   ```

5. Track the checkpoint with Git LFS and push:
   ```bash
   git lfs install
   git lfs track "model.pt"
   git add .gitattributes Dockerfile README.md model.pt api/ app/ inference/ model/ tokenizer/
   git commit -m "Deploy MiniGPT-X Docker Space"
   git push origin main
   ```

6. Open `https://huggingface.co/spaces/<YOUR_USERNAME>/minigpt-x` to view the live story generator.

---

## 2. Publish Project to GitHub

1. Create a new repository on GitHub named `minigpt-x` (do not initialize with README or license).

2. Link your local project and push:
   ```bash
   cd /Users/praveenchourasia/Projects/minigpt-x
   git remote add origin https://github.com/<YOUR_USERNAME>/minigpt-x.git
   git branch -M main
   git push -u origin main
   ```

---

## 3. Upload Weights & Model Card to Hugging Face Hub (Optional)

1. Create a model repository on Hugging Face: [https://huggingface.co/new](https://huggingface.co/new) (select Model).
2. Upload `deploy/hf_space/model.pt` and `MODEL_CARD.md`.

