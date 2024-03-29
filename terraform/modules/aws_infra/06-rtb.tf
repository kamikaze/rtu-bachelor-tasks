# Route table

resource "aws_default_route_table" "default_rtb" {
  route {
    ipv6_cidr_block = "::/0"
    gateway_id      = aws_internet_gateway.main_internet_gateway.id
  }

  default_route_table_id = aws_vpc.main_vpc.default_route_table_id

  tags = {
    Name         = "${var.project}-default-rtb",
    Organization = var.organization
    Project      = var.project
  }
}


resource "aws_route_table" "private_rtb" {
  depends_on = [
    aws_egress_only_internet_gateway.egress_only_internet_gateway,
    aws_nat_gateway.nat
  ]

  vpc_id = aws_vpc.main_vpc.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat.id
  }

  route {
    ipv6_cidr_block        = "::/0"
    egress_only_gateway_id = aws_egress_only_internet_gateway.egress_only_internet_gateway.id
  }

  tags = {
    Name         = "${var.project}-private-rtb",
    Organization = var.organization
    Project      = var.project
  }
}

resource "aws_route_table" "public_rtb" {
  depends_on = [
    aws_internet_gateway.main_internet_gateway
  ]
  vpc_id = aws_vpc.main_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main_internet_gateway.id
  }

  route {
    ipv6_cidr_block = "::/0"
    gateway_id      = aws_internet_gateway.main_internet_gateway.id
  }

  tags = {
    Name         = "${var.project}-public-rtb",
    Organization = var.organization
    Project      = var.project
  }
}

resource "aws_route_table_association" "private-a" {
  depends_on = [
    aws_nat_gateway.nat,
    aws_route_table.private_rtb
  ]
  route_table_id = aws_route_table.private_rtb.id
  subnet_id      = aws_subnet.private_subnet_a.id
}

resource "aws_route_table_association" "private-b" {
  depends_on = [
    aws_nat_gateway.nat,
    aws_route_table.private_rtb
  ]
  route_table_id = aws_route_table.private_rtb.id
  subnet_id      = aws_subnet.private_subnet_b.id
}

resource "aws_route_table_association" "private-c" {
  depends_on = [
    aws_nat_gateway.nat,
    aws_route_table.private_rtb
  ]
  route_table_id = aws_route_table.private_rtb.id
  subnet_id      = aws_subnet.private_subnet_c.id
}

resource "aws_route_table_association" "public-a" {
  depends_on = [
    aws_internet_gateway.main_internet_gateway,
    aws_route_table.public_rtb
  ]
  route_table_id = aws_route_table.public_rtb.id
  subnet_id      = aws_subnet.public_subnet_a.id
}

resource "aws_route_table_association" "public-b" {
  depends_on = [
    aws_internet_gateway.main_internet_gateway,
    aws_route_table.public_rtb
  ]
  route_table_id = aws_route_table.public_rtb.id
  subnet_id      = aws_subnet.public_subnet_b.id
}

resource "aws_route_table_association" "public-c" {
  depends_on = [
    aws_internet_gateway.main_internet_gateway,
    aws_route_table.public_rtb
  ]
  route_table_id = aws_route_table.public_rtb.id
  subnet_id      = aws_subnet.public_subnet_c.id
}