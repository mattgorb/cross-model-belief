# Terraform — spot GPU for cross-model belief experiments
#
# Usage:
#   terraform init
#   terraform apply -var 'key_name=YOUR_KEY' -var 'my_ip=1.2.3.4/32'
# Dev / <=8B models: default g5.2xlarge (1x A10G 24GB).
# 32B fp16 sharded:   -var 'instance_type=g5.12xlarge' (4x A10G 96GB).
#
# The gp3 EBS volume holds the HF model cache and the activation cache. With a
# one-time spot request, detach/reattach this volume to a fresh instance to keep
# the expensive 32B activation extraction across spot interruptions.

terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" {
  region = var.region
}

variable "region"        { default = "us-east-1" }
variable "instance_type" { default = "g5.2xlarge" } # bump to g5.12xlarge for 32B
variable "key_name"      {}                          # your EC2 keypair name
variable "my_ip"         {}                          # e.g. "1.2.3.4/32" for SSH
variable "max_spot_price" { default = "3.00" }
variable "volume_gb"     { default = 300 }

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
  ingress {
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

resource "aws_instance" "gpu" {
  ami                    = data.aws_ami.dlami.id
  instance_type          = var.instance_type
  key_name               = var.key_name
  vpc_security_group_ids = [aws_security_group.gpu.id]

  instance_market_options {
    market_type = "spot"
    spot_options {
      max_price          = var.max_spot_price
      spot_instance_type = "one-time"
    }
  }

  root_block_device {
    volume_size = var.volume_gb # HF cache + activations
    volume_type = "gp3"
  }

  tags = { Name = "cross-model-probe" }
}

output "ssh" {
  value = "ssh ubuntu@${aws_instance.gpu.public_ip}"
}
