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
