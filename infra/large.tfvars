# Tier 2 — the big models (qwen3-32b ~64GB, qwen38-27b ~54GB, gemma4-12b ~24GB).
# 4x A10G 96GB, sharded fp16.
#   terraform destroy -target=aws_instance.gpu    # drop the small box first
#   terraform apply -var-file=large.tfvars        # same volume, bigger box
instance_type  = "g5.12xlarge"
data_volume_gb = 300
max_spot_price = "3.00"

# On-demand: spot quota (L-3819A6DF) is 0 until AWS raises it. g5.12xlarge
# on-demand is ~$5.67/hr, so extract deliberately and destroy the box after.
use_spot = false
