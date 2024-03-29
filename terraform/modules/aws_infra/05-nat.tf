resource "aws_eip" "nat" {
  domain = "vpc"

  tags = {
    Name         = "${var.project}-nat-eip"
    Organization = var.organization
    Project      = var.project
  }
}

resource "aws_nat_gateway" "nat" {
  depends_on = [
    aws_internet_gateway.main_internet_gateway,
    aws_eip.nat
  ]

  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public_subnet_a.id

  tags = {
    Name         = "${var.project}-nat"
    Organization = var.organization
    Project      = var.project
  }
}