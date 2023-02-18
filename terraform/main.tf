terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.55"
    }
  }

  required_version = ">= 1.2.0"
}

provider "aws" {
  region  = "eu-north-1"
  profile = "rtu"
}

# VPC

resource "aws_vpc" "rtu_bachelor_vpc" {
  cidr_block = "10.0.0.0/16"
}

# Security groups

resource "aws_security_group" "rtu_bachelor_ssh_sg" {
  depends_on  = [aws_vpc.rtu_bachelor_vpc]
  name        = "rtu-bachelor-ssh-sg"
  description = "SSH security group"
#  vpc_id      = aws_vpc.rtu_bachelor_vpc.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "rtu_bachelor_postgres_sg" {
  depends_on  = [aws_vpc.rtu_bachelor_vpc]
  name        = "rtu-bachelor-postgres-sg"
  description = "PostgreSQL security group"
#  vpc_id      = aws_vpc.rtu_bachelor_vpc.id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# S3 buckets

resource "aws_s3_bucket" "rtu_dataset_transport_raw" {
  bucket = "rtu-dataset-transport-raw"

  tags = {
    Name = "RTU Dataset: Transport RAW"
  }
}

resource "aws_s3_bucket_acl" "rtu_dataset_transport_raw_acl" {
  bucket = aws_s3_bucket.rtu_dataset_transport_raw.id
  acl    = "private"
}

resource "aws_s3_bucket" "rtu_dataset_transport_gen" {
  bucket = "rtu-dataset-transport-gen"

  tags = {
    Name = "RTU Dataset: Transport generated"
  }
}

resource "aws_s3_bucket_acl" "rtu_dataset_transport_gen_acl" {
  bucket = aws_s3_bucket.rtu_dataset_transport_gen.id
  acl    = "private"
}

resource "aws_s3_bucket" "rtu_dataset_transport_thumb" {
  bucket = "rtu-dataset-transport-thumb"

  tags = {
    Name = "RTU Dataset: Transport thumbnails"
  }
}

resource "aws_s3_bucket_acl" "rtu_dataset_transport_thumb_acl" {
  bucket = aws_s3_bucket.rtu_dataset_transport_thumb.id
  acl    = "public-read"
}

# EC2 instances

resource "aws_spot_instance_request" "k3s_master_spot_ec2_instance" {
  depends_on                     = [aws_security_group.rtu_bachelor_ssh_sg]
  ami                            = "ami-0c76bf0e69c8a6228"
  spot_price                     = "0.01"
  spot_type                      = "persistent"
  instance_type                  = "t4g.small"
  instance_interruption_behavior = "hibernate"
  wait_for_fulfillment           = "true"
  key_name                       = "cl-dev-keypair"
  vpc_security_group_ids         = [aws_security_group.rtu_bachelor_ssh_sg.id]

  tags = {
    Arch      = "arm64"
    HasGPU    = "no"
    Name      = "k3s-master1"
    SpotPrice = "yes"
  }
}

resource "aws_spot_instance_request" "rtu_cpu_spot_ec2_instance" {
  depends_on = [
    aws_security_group.rtu_bachelor_ssh_sg,
    aws_spot_instance_request.k3s_master_spot_ec2_instance
  ]
  ami                            = "ami-0c76bf0e69c8a6228"
  spot_price                     = "1.00"
  spot_type                      = "persistent"
  instance_type                  = "c6g.16xlarge"
  instance_interruption_behavior = "hibernate"
  wait_for_fulfillment           = "true"
  key_name                       = "cl-dev-keypair"
  vpc_security_group_ids         = [aws_security_group.rtu_bachelor_ssh_sg.id]

  tags = {
    Arch      = "arm64"
    HasGPU    = "no"
    Name      = "rtu-cpu-monster1"
    SpotPrice = "yes"
  }
}

#resource "aws_spot_instance_request" "rtu_gpu_spot_ec2_instance" {
#  depends_on = [
#    aws_security_group.rtu_bachelor_ssh_sg,
#    aws_spot_instance_request.k3s_master_spot_ec2_instance
#  ]
#  ami                            = "ami-0bb935e4614c12d86"
#  spot_price                     = "1.75"
#  spot_type                      = "persistent"
#  instance_type                  = "g5.16xlarge"
#  instance_interruption_behavior = "hibernate"
#  wait_for_fulfillment           = "true"
#  key_name                       = "cl-dev-keypair"
#  vpc_security_group_ids         = [aws_security_group.rtu_bachelor_ssh_sg.id]
#
#  tags = {
#    Arch      = "amd64"
#    HasGPU    = "yes"
#    Name      = "rtu-gpu-monster1"
#    SpotPrice = "yes"
#  }
#}

# Outputs

output "k3s_master_ip" {
  value = aws_spot_instance_request.k3s_master_spot_ec2_instance.public_ip
}

output "rtu_cpu_monster_ip" {
  value = aws_spot_instance_request.rtu_cpu_spot_ec2_instance.public_ip
}

#output "rtu_gpu_monster_ip" {
#  value = aws_spot_instance_request.rtu_gpu_spot_ec2_instance.public_ip
#}
