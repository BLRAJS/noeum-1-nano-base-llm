<p align="center">
  <img src="https://noeum.ai/wp-content/uploads/2025/11/noeum.png" alt="Noeum.ai" width="140" />
</p>

# Noeum-1-Nano-Base LLM — Foundation Checkpoint (From-Scratch MoE)

This repository provides a **minimal, practical starter** for **Noeum-1-Nano-Base** — the **raw pre-trained** (foundation) checkpoint intended for:
- **text completion** and **few-shot learning**
- **fine-tuning** into domain or instruction models
- lightweight evaluation prompts and baseline scripts

> ⚠️ **Base model notice:** This is a **pre-trained base model** (no SFT, no RLHF/DPO).  
> For chat + reasoning behavior, use the post-trained release:  
> https://huggingface.co/noeum/noeum-1-nano

**Links**
- Website: https://noeum.ai  
- Base model (HF): **[ADD BASE MODEL LINK HERE]**  
- Post-trained model (HF): https://huggingface.co/noeum/noeum-1-nano  

---

## About Noeum.ai

**Noeum.ai** is an independent AI research & engineering lab based in **Vienna, Austria**, focused on **efficiency-first training** and **reproducible evaluation**.  
Founded and led by **Bledar Ramo**, the lab builds models end-to-end—from pre-training to post-training and benchmarking—with the goal of improving reasoning quality **per unit of compute** and advancing **novel reasoning capabilities**.

**Core philosophy:** iterate fast at nano scale, validate what works, then scale only proven techniques.

---

## What is Noeum-1-Nano-Base?

**Noeum-1-Nano-Base** is a nano-scale **Mixture-of-Experts (MoE)** foundation language model:
- **Size:** 0.6B total parameters / ~0.2B active parameters  
- **Training:** 18B high-signal tokens  
- **From scratch:** no pretrained weights, no inherited checkpoints  
- **Purpose:** a clean, efficient starting point for **completion** and **fine-tuning**

This repo is intentionally lightweight: it exists to help users **load, test, and build on** the base checkpoint quickly.

---
