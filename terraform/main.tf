variable "db_username" {
  type        = string
  nullable    = false
  default     = "bachelor"
  description = "Username for a postgres database"
}

variable "db_password" {
  type        = string
  nullable    = false
  default     = "bachelorbachelor18"
  description = "Password for a postgres database"

  validation {
    condition     = length(var.db_password) >= 18
    error_message = "Database password must be at least 18 characters long."
  }
}

variable "eks-cluster-name" {
  type    = string
  default = "rtu-bachelor-eks"
}


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

  tags = {
    Name         = "rtu-bachelor-vpc",
    Organization = "RTU"
  }
}

# Internet gateway

resource "aws_internet_gateway" "rtu_bachelor_internet_gateway" {
  depends_on = [
    aws_vpc.rtu_bachelor_vpc
  ]

  vpc_id = aws_vpc.rtu_bachelor_vpc.id
  tags = {
    Name         = "rtu-bachelor-igw",
    Organization = "RTU"
  }
}

# Route table

resource "aws_default_route_table" "rtu_bachelor_default_rtb" {
  depends_on = [
    aws_vpc.rtu_bachelor_vpc
  ]

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.rtu_bachelor_internet_gateway.id
  }

  default_route_table_id = aws_vpc.rtu_bachelor_vpc.default_route_table_id

  tags = {
    Name         = "rtu-bachelor-default-rtb",
    Organization = "RTU"
  }
}

# Subnets

resource "aws_subnet" "rtu_bachelor_subnet_eu_north_1a" {
  depends_on = [
    aws_vpc.rtu_bachelor_vpc,
    aws_default_route_table.rtu_bachelor_default_rtb
  ]

  vpc_id            = aws_vpc.rtu_bachelor_vpc.id
  availability_zone = "eu-north-1a"
  cidr_block        = "10.0.1.0/24"

  tags = {
    Name         = "rtu-bachelor-subnet-eu-north-1a",
    Organization = "RTU"
  }
}

resource "aws_subnet" "rtu_bachelor_subnet_eu_north_1b" {
  depends_on = [
    aws_vpc.rtu_bachelor_vpc,
    aws_default_route_table.rtu_bachelor_default_rtb
  ]

  vpc_id            = aws_vpc.rtu_bachelor_vpc.id
  availability_zone = "eu-north-1b"
  cidr_block        = "10.0.2.0/24"

  tags = {
    Name         = "rtu-bachelor-subnet-eu-north-1b",
    Organization = "RTU"
  }
}

resource "aws_subnet" "rtu_bachelor_subnet_eu_north_1c" {
  depends_on = [
    aws_vpc.rtu_bachelor_vpc,
    aws_default_route_table.rtu_bachelor_default_rtb
  ]

  vpc_id            = aws_vpc.rtu_bachelor_vpc.id
  availability_zone = "eu-north-1c"
  cidr_block        = "10.0.3.0/24"

  tags = {
    Name         = "rtu-bachelor-subnet-eu-north-1c",
    Organization = "RTU"
  }
}

# Security groups

resource "aws_security_group" "rtu_bachelor_ssh_sg" {
  depends_on = [
    aws_vpc.rtu_bachelor_vpc
  ]
  name        = "rtu-bachelor-ssh-sg"
  description = "SSH security group"
  vpc_id      = aws_vpc.rtu_bachelor_vpc.id

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

  tags = {
    Name         = "rtu-bachelor-ssh-sg",
    Organization = "RTU"
  }
}

resource "aws_security_group" "rtu_bachelor_postgres_sg" {
  depends_on = [
    aws_vpc.rtu_bachelor_vpc
  ]
  name        = "rtu-bachelor-postgres-sg"
  description = "PostgreSQL security group"
  vpc_id      = aws_vpc.rtu_bachelor_vpc.id

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

  tags = {
    Name         = "rtu-bachelor-postgres-sg",
    Organization = "RTU"
  }
}

# S3 buckets

resource "aws_s3_bucket" "rtu_dataset_transport_raw" {
  bucket = "rtu-dataset-transport-raw"

  #  lifecycle {
  #    prevent_destroy = true
  #  }

  tags = {
    Name         = "RTU Dataset: Transport RAW",
    Organization = "RTU"
  }
}

resource "aws_s3_bucket_acl" "rtu_dataset_transport_raw_acl" {
  bucket = aws_s3_bucket.rtu_dataset_transport_raw.id
  acl    = "private"
}

resource "aws_s3_bucket_public_access_block" "rtu_dataset_transport_raw_pab" {
  bucket = aws_s3_bucket.rtu_dataset_transport_raw.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket" "rtu_dataset_transport_gen" {
  bucket = "rtu-dataset-transport-gen"

  #  lifecycle {
  #    prevent_destroy = true
  #  }

  tags = {
    Name         = "RTU Dataset: Transport generated",
    Organization = "RTU"
  }
}

resource "aws_s3_bucket_acl" "rtu_dataset_transport_gen_acl" {
  bucket = aws_s3_bucket.rtu_dataset_transport_gen.id
  acl    = "private"
}

resource "aws_s3_bucket_public_access_block" "rtu_dataset_transport_gen_pab" {
  bucket = aws_s3_bucket.rtu_dataset_transport_gen.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket" "rtu_dataset_transport_thumb" {
  bucket = "rtu-dataset-transport-thumb"

  #  lifecycle {
  #    prevent_destroy = true
  #  }

  tags = {
    Name         = "RTU Dataset: Transport thumbnails",
    Organization = "RTU"
  }
}

resource "aws_s3_bucket_acl" "rtu_dataset_transport_thumb_acl" {
  bucket = aws_s3_bucket.rtu_dataset_transport_thumb.id
  acl    = "public-read"
}

# ECR

resource "aws_ecr_repository" "rtu_dataset_converter" {
  name = "rtu-dataset-converter"

  #  lifecycle {
  #    prevent_destroy = true
  #  }

  tags = {
    Name         = "rtu-dataset-converter"
    Organization = "RTU"
  }
}

# EC2 instances

resource "aws_spot_instance_request" "k3s_master_spot_ec2_instance" {
  depends_on = [
    aws_subnet.rtu_bachelor_subnet_eu_north_1a,
    aws_security_group.rtu_bachelor_ssh_sg
  ]
  ami                            = "ami-0c76bf0e69c8a6228"
  spot_price                     = "0.01"
  spot_type                      = "persistent"
  instance_type                  = "t4g.small"
  instance_interruption_behavior = "stop"
  wait_for_fulfillment           = true
  key_name                       = "cl-dev-keypair"
  subnet_id                      = aws_subnet.rtu_bachelor_subnet_eu_north_1a.id
  vpc_security_group_ids         = [aws_security_group.rtu_bachelor_ssh_sg.id]
  associate_public_ip_address    = true

  tags = {
    Arch         = "arm64"
    HasGPU       = "no"
    Name         = "k3s-master1",
    Organization = "RTU"
    SpotPrice    = "yes"
  }
}

#resource "aws_spot_instance_request" "rtu_cpu_spot_ec2_instance" {
#  depends_on = [
#    aws_subnet.rtu_bachelor_subnet_eu_north_1a,
#    aws_security_group.rtu_bachelor_ssh_sg,
#    aws_spot_instance_request.k3s_master_spot_ec2_instance
#  ]
#  ami                            = "ami-0c76bf0e69c8a6228"
#  spot_price                     = "1.00"
#  spot_type                      = "persistent"
#  instance_type                  = "c6g.16xlarge"
#  instance_interruption_behavior = "stop"
#  wait_for_fulfillment           = true
#  key_name                       = "cl-dev-keypair"
#  subnet_id                      = aws_subnet.rtu_bachelor_subnet_eu_north_1a.id
#  vpc_security_group_ids         = [aws_security_group.rtu_bachelor_ssh_sg.id]
#  associate_public_ip_address    = true
#
#  tags = {
#    Arch         = "arm64"
#    HasGPU       = "no"
#    Name         = "rtu-cpu-monster1",
#    Organization = "RTU"
#    SpotPrice    = "yes"
#  }
#}

#resource "aws_spot_instance_request" "rtu_gpu_spot_ec2_instance" {
#  depends_on = [
#    aws_subnet.rtu_bachelor_subnet_eu_north_1a,
#    aws_security_group.rtu_bachelor_ssh_sg,
#    aws_spot_instance_request.k3s_master_spot_ec2_instance
#  ]
#  ami                            = "ami-0bb935e4614c12d86"
#  spot_price                     = "1.75"
#  spot_type                      = "persistent"
#  instance_type                  = "g5.16xlarge"
#  instance_interruption_behavior = "stop"
#  wait_for_fulfillment           = true
#  key_name                       = "cl-dev-keypair"
#  subnet_id                      = aws_subnet.rtu_bachelor_subnet_eu_north_1a.id
#  vpc_security_group_ids         = [aws_security_group.rtu_bachelor_ssh_sg.id]
#  associate_public_ip_address    = true
#
#  tags = {
#    Arch         = "amd64"
#    HasGPU       = "yes"
#    Name         = "rtu-gpu-monster1",
#    Organization = "RTU"
#    SpotPrice    = "yes"
#  }
#}

# RDS subnet

resource "aws_db_subnet_group" "rtu_bachelor_db_subnet_group" {
  depends_on = [
    aws_subnet.rtu_bachelor_subnet_eu_north_1a,
    aws_subnet.rtu_bachelor_subnet_eu_north_1b,
    aws_subnet.rtu_bachelor_subnet_eu_north_1c,
  ]
  name = "rtu-bachelor-db-subnet-group"
  subnet_ids = [
    aws_subnet.rtu_bachelor_subnet_eu_north_1a.id,
    aws_subnet.rtu_bachelor_subnet_eu_north_1b.id,
    aws_subnet.rtu_bachelor_subnet_eu_north_1c.id
  ]
}

# RDS

resource "aws_db_instance" "rtu_bachelor_db_master" {
  depends_on = [
    aws_security_group.rtu_bachelor_postgres_sg,
    aws_db_subnet_group.rtu_bachelor_db_subnet_group
  ]

  identifier              = "rtu-bachelor-db"
  engine                  = "postgres"
  engine_version          = "14.6"
  db_name                 = "datasets"
  instance_class          = "db.t4g.micro"
  storage_type            = "gp2"
  allocated_storage       = 10
  backup_retention_period = 7
  availability_zone       = "eu-north-1a"
  maintenance_window      = "Sun:04:00-Sun:06:00"
  backup_window           = "02:00-03:00"
  apply_immediately       = true
  db_subnet_group_name    = aws_db_subnet_group.rtu_bachelor_db_subnet_group.name
  vpc_security_group_ids  = [aws_security_group.rtu_bachelor_postgres_sg.id]
  username                = var.db_username
  password                = var.db_password
  skip_final_snapshot     = true

  tags = {
    Organization = "RTU"
  }
}

#locals {
#  replicas = [1, 2, 3]
#  azs      = ["a", "b", "c"]
#}
#
#resource "aws_db_instance" "rtu_bachelor_db_ro_replica" {
#  depends_on = [
#    aws_security_group.rtu_bachelor_postgres_sg,
#    aws_db_subnet_group.rtu_bachelor_db_subnet_group,
#    aws_db_instance.rtu_bachelor_db_master
#  ]
#  count = length(local.replicas)
#
#  identifier              = "rtu-bachelor-db-ro${local.replicas[count.index]}"
#  instance_class          = "db.t4g.micro"
#  storage_type            = "gp2"
#  backup_retention_period = 7
#  availability_zone       = "eu-north-1${local.azs[count.index]}"
#  replicate_source_db     = aws_db_instance.rtu_bachelor_db_master.id
#  maintenance_window      = "Sun:04:00-Sun:06:00"
#  backup_window           = "02:00-03:00"
#  apply_immediately       = true
#  vpc_security_group_ids  = [aws_security_group.rtu_bachelor_postgres_sg.id]
#  skip_final_snapshot     = true
#
#  tags = {
#    Organization = "RTU"
#  }
#}

# Outputs

output "ecr_repository_url" {
  value = aws_ecr_repository.rtu_dataset_converter.repository_url
}

output "k3s_master_ip" {
  value = aws_spot_instance_request.k3s_master_spot_ec2_instance.public_ip
}

#output "rtu_cpu_monster_ip" {
#  value = aws_spot_instance_request.rtu_cpu_spot_ec2_instance.public_ip
#}

#output "rtu_gpu_monster_ip" {
#  value = aws_spot_instance_request.rtu_gpu_spot_ec2_instance.public_ip
#}

output "rtu_bachelor_db_master_ip" {
  value = aws_db_instance.rtu_bachelor_db_master.address
}

#output "rtu_bachelor_db_ro_replica_ip" {
#  value = aws_db_instance.rtu_bachelor_db_ro_replica[*].address
#}
