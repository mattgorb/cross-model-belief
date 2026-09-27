# Tier 1 — models up to 8B (qwen3-1.7b, qwen3-4b, qwen3-8b, llama-8b).
# 1x A10G 24GB. 8B in fp16 is ~16GB, so this fits with headroom.
#   terraform apply -var-file=small.tfvars
instance_type  = "g5.xlarge"
data_volume_gb = 100
max_spot_price = "1.00"

# Spot quota (L-3819A6DF) is 0 until AWS raises it, so start on-demand (~$1.01/hr).
use_spot = false
