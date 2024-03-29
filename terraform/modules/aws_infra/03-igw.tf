# Internet gateway

resource "aws_internet_gateway" "main_internet_gateway" {
  vpc_id = aws_vpc.main_vpc.id

  tags = {
    Name         = "${var.project}-igw",
    Organization = var.organization
    Project      = var.project
  }
}

resource "aws_egress_only_internet_gateway" "egress_only_internet_gateway" {
  vpc_id = aws_vpc.main_vpc.id

  tags = {
    Name         = "${var.project}-eo-igw",
    Organization = var.organization
    Project      = var.project
  }
}