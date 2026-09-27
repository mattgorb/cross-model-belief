# Terraform — spot GPU for cross-model belief experiments
#
#   terraform init
#   terraform apply -var "key_name=YOUR_KEY" -var "my_ip=$(curl -s ifconfig.me)/32"
#   terraform output -raw ssh_command
#
# Two resources matter and they have different lifetimes:
#
#   aws_instance.gpu    — the spot box. Cheap to lose, cheap to recreate.
#   aws_ebs_volume.data — /mnt/data, holding the HF weights and the activation
#                         cache. This is the expensive thing: the 32B extraction
#                         pass lives here. It is deliberately a *separate*
#                         volume, not the root device, so a spot interruption
#                         costs the instance and not the vectors.
#
# To replace the box and keep the data:
#   terraform destroy -target=aws_instance.gpu          # volume survives
#   terraform apply -var ... -var "instance_type=g5.12xlarge"
#
# To reuse a volume across separate Terraform states (or after a full destroy),
# pass its id: -var "existing_data_volume_id=vol-0123...".
#
# Sizing: g5.2xlarge (1x A10G 24GB) covers everything up to 8B; the 32B fp16
# pass needs g5.12xlarge (4x A10G 96GB).

terraform {
  required_version = ">= 1.3"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" {
  region = var.region
}

variable "region" {
  description = "AWS region."
  default     = "us-east-1"
}

variable "instance_type" {
  description = "Set via -var-file: small.tfvars (<=8B) or large.tfvars (27B/32B)."
  default     = "g5.xlarge"
}

variable "key_name" {
  description = "Name of an existing EC2 keypair (for SSH)."
}

variable "my_ip" {
  description = "Your IP in CIDR form, e.g. \"1.2.3.4/32\". SSH is open to this only."
}

variable "use_spot" {
  description = "Spot needs a nonzero 'All G and VT Spot Instance Requests' quota (L-3819A6DF); it is 0 by default on new accounts. false = on-demand, ~2-3x the price but launches immediately."
  type        = bool
  default     = true
}

variable "max_spot_price" {
  description = "Spot ceiling, USD/hour. On-demand is ~1.21 (2xlarge) / ~5.67 (12xlarge)."
  default     = "3.00"
}

variable "root_volume_gb" {
  description = "Root disk. The DLAMI itself needs ~100GB; nothing of ours goes here."
  default     = 150
}

variable "data_volume_gb" {
  description = "The /mnt/data volume: HF weights + activation cache. ~350MB per 7B model-dataset at N=2000, plus ~60GB of weights for the 32B."
  default     = 100
}

variable "existing_data_volume_id" {
  description = "Reattach an existing volume (vol-...) instead of creating one. Empty means create."
  default     = ""
}

variable "availability_zone" {
  description = "AZ for both instance and volume — they must match. Empty picks the first available."
  default     = ""
}

data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  az = var.availability_zone != "" ? var.availability_zone : data.aws_availability_zones.available.names[0]
  # Created volume, or the one passed in — whichever exists.
  data_volume_id = var.existing_data_volume_id != "" ? var.existing_data_volume_id : aws_ebs_volume.data[0].id
}

# Default VPC subnet in the chosen AZ, so no networking has to be invented.
resource "aws_default_vpc" "default" {}

resource "aws_default_subnet" "default" {
  availability_zone = local.az

  # Nothing here references the VPC, so without this Terraform creates both in
  # parallel and CreateDefaultSubnet fails with DefaultVpcDoesNotExist.
  depends_on = [aws_default_vpc.default]
}

data "aws_ami" "dlami" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu 22.04*"]
  }
}

resource "aws_security_group" "gpu" {
  name        = "cross-model-probe-sg"
  description = "SSH from my IP only"
  vpc_id      = aws_default_vpc.default.id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.my_ip]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# The volume that must outlive the instance.
resource "aws_ebs_volume" "data" {
  count             = var.existing_data_volume_id != "" ? 0 : 1
  availability_zone = local.az
  size              = var.data_volume_gb
  type              = "gp3"
  encrypted         = true
  tags              = { Name = "cross-model-probe-data" }

  # The 32B extraction is hours of GPU time. Deleting this volume by accident is
  # the one unrecoverable mistake in the whole setup, so Terraform refuses to.
  # To remove it deliberately: comment this out, or delete it in the console.
  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_instance" "gpu" {
  ami                    = data.aws_ami.dlami.id
  instance_type          = var.instance_type
  key_name               = var.key_name
  subnet_id              = aws_default_subnet.default.id
  vpc_security_group_ids = [aws_security_group.gpu.id]

  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        max_price = var.max_spot_price
        # one-time: an interrupted instance stays gone rather than coming back
        # half-configured. Re-run apply; the data volume is still there.
        spot_instance_type = "one-time"
      }
    }
  }

  root_block_device {
    volume_size = var.root_volume_gb
    volume_type = "gp3"
  }

  # Format on first use only, then mount. Idempotent: a volume that already
  # holds a filesystem (a reattached one) is mounted, never reformatted.
  #
  # The device is located by *volume id*, not by guessing a device name. On
  # Nitro instances (g5 included) the EBS volume shows up as some /dev/nvmeXn1
  # whose number is not stable, and on g5 /dev/nvme1n1 is typically the
  # ephemeral instance store — formatting that would put the activation cache on
  # a disk that dies with the spot instance, which is the exact failure this
  # volume exists to prevent.
  user_data = <<-SCRIPT
    #!/bin/bash
    set -euxo pipefail
    SERIAL="$(echo "${local.data_volume_id}" | tr -d '-')"
    BY_ID="/dev/disk/by-id/nvme-Amazon_Elastic_Block_Store_$SERIAL"
    DEV=""
    for i in $(seq 1 60); do
      for c in "$BY_ID" /dev/sdf /dev/xvdf; do
        if [ -b "$c" ]; then DEV="$(readlink -f "$c")"; break; fi
      done
      [ -n "$DEV" ] && break
      sleep 5
    done
    [ -n "$DEV" ] || { echo "data volume ${local.data_volume_id} never appeared" >&2; exit 1; }
    blkid "$DEV" || mkfs -t ext4 "$DEV"
    mkdir -p /mnt/data
    UUID="$(blkid -s UUID -o value "$DEV")"
    grep -q "$UUID" /etc/fstab || echo "UUID=$UUID /mnt/data ext4 defaults,nofail 0 2" >> /etc/fstab
    mount -a
    mkdir -p /mnt/data/hf /mnt/data/activations
    chown -R ubuntu:ubuntu /mnt/data
    # So an interactive shell inherits the right cache paths without being told.
    cat >> /home/ubuntu/.bashrc <<'ENV'
    export HF_HOME=/mnt/data/hf
    export CMB_CACHE=/mnt/data/activations
    ENV
  SCRIPT

  tags = { Name = "cross-model-probe" }
}

resource "aws_volume_attachment" "data" {
  device_name = "/dev/sdf"
  volume_id   = local.data_volume_id
  instance_id = aws_instance.gpu.id

  # Let the instance go away without Terraform waiting on a clean detach — the
  # point of the separate volume is that losing the box is routine.
  skip_destroy = true
}

output "ssh_command" {
  description = "Copy-paste to get on the box."
  value       = "ssh ubuntu@${aws_instance.gpu.public_ip}"
}

output "public_ip" {
  value = aws_instance.gpu.public_ip
}

output "instance_type" {
  value = aws_instance.gpu.instance_type
}

output "data_volume_id" {
  description = "Pass this as -var existing_data_volume_id=... to reattach it later."
  value       = local.data_volume_id
}

output "availability_zone" {
  value = local.az
}
